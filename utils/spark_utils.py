"""
Spark utility functions for the Numerai Crypto project.
"""
import os
import sys
import psutil
from pyspark.sql import SparkSession
from pysparkling import H2OContext
import h2o
from numer_crypto.config.settings import SPARK_CONFIG, H2O_CONFIG, HARDWARE_CONFIG


def get_system_resources():
    """
    Detect available system resources for dynamic configuration.
    
    Returns:
        dict: Current system resources (memory, cores, GPUs)
    """
    resources = {}
    
    # Get available memory
    mem_info = psutil.virtual_memory()
    resources['available_memory_gb'] = mem_info.total / (1024**3)
    
    # Get available CPU cores
    resources['cpu_count'] = psutil.cpu_count(logical=True)
    resources['physical_cpu_count'] = psutil.cpu_count(logical=False)
    
    # Get GPU information from hardware config
    resources['gpu_count'] = HARDWARE_CONFIG.get('gpu_count', 0)
    resources['gpu_memory'] = HARDWARE_CONFIG.get('gpu_memory', '0g')
    
    print(f"System resources detected: {resources}")
    return resources


def create_spark_session(dynamic_config=True):
    """
    Create and return a SparkSession with configured settings.
    
    Args:
        dynamic_config (bool): Whether to dynamically adjust configuration based on available resources
    
    Returns:
        SparkSession: Configured Spark session
    """
    # Set Java home if needed
    if 'JAVA_HOME' not in os.environ:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
    
    # Create builder
    builder = SparkSession.builder.appName(SPARK_CONFIG['app_name'])
    
    # Apply all configurations from settings
    for key, value in SPARK_CONFIG.items():
        if key != 'app_name' and not key.startswith('spark.'):
            builder = builder.config(f"spark.{key}", value)
        elif key.startswith('spark.'):
            builder = builder.config(key, value)
    
    # Dynamic configuration based on available resources
    if dynamic_config:
        resources = get_system_resources()
        
        # Adjust memory settings based on available memory
        avail_mem_gb = resources['available_memory_gb']
        if avail_mem_gb > 600:  # Close to 640GB
            # We have our full 640GB available
            driver_mem = '64g'
            executor_mem = '128g'
            max_result_size = '16g'
        elif avail_mem_gb > 300:  # Over half of intended memory
            driver_mem = '32g'
            executor_mem = '64g'
            max_result_size = '8g'
        elif avail_mem_gb > 100:  # Decent amount of memory
            driver_mem = '16g'
            executor_mem = '32g'
            max_result_size = '4g'
        else:  # Limited memory
            driver_mem = '4g'
            executor_mem = '8g'
            max_result_size = '2g'
        
        builder = builder.config("spark.driver.memory", driver_mem)
        builder = builder.config("spark.executor.memory", executor_mem)
        builder = builder.config("spark.driver.maxResultSize", max_result_size)
        
        # Adjust parallelism based on CPU cores
        if resources['cpu_count'] > 32:
            builder = builder.config("spark.default.parallelism", "200")
            builder = builder.config("spark.sql.shuffle.partitions", "2000")
        elif resources['cpu_count'] > 16:
            builder = builder.config("spark.default.parallelism", "100")
            builder = builder.config("spark.sql.shuffle.partitions", "1000")
            
        # Set master to local with all cores
        builder = builder.config("spark.master", f"local[{resources['cpu_count']}]")
    
    # Create the session
    spark = builder.getOrCreate()
    
    # Log configuration 
    print("Spark Configuration:")
    for entry in spark.sparkContext.getConf().getAll():
        print(f"  {entry[0]}: {entry[1]}")
        
    return spark


def init_h2o(dynamic_config=True):
    """
    Initialize H2O and return the H2O instance.
    
    Args:
        dynamic_config (bool): Whether to dynamically adjust configuration based on available resources
    
    Returns:
        h2o_context: Initialized H2O instance
    """
    if dynamic_config:
        resources = get_system_resources()
        
        # Determine memory size based on available resources
        avail_mem_gb = resources['available_memory_gb']
        if avail_mem_gb > 600:  # Close to 640GB
            mem_size = '256g'
        elif avail_mem_gb > 300:  # Over half of intended memory
            mem_size = '128g'
        elif avail_mem_gb > 100:  # Decent amount of memory
            mem_size = '64g'
        else:  # Limited memory
            mem_size = '4g'
            
        # Determine GPU usage
        if resources['gpu_count'] > 0:
            backend = 'gpu'
            gpu_ids = list(range(min(resources['gpu_count'], 3)))  # Up to 3 GPUs
        else:
            backend = None
            gpu_ids = None
            
        # Initialize with dynamic configuration
        init_args = {
            'max_mem_size': mem_size,
            'nthreads': -1,  # Use all threads
        }
        
        if backend == 'gpu':
            init_args['backend'] = backend
            if gpu_ids:
                init_args['gpu_id'] = gpu_ids
                
        print(f"Initializing H2O with settings: {init_args}")
        h2o.init(**init_args)
    else:
        # Use configuration from settings
        init_args = {}
        for key, value in H2O_CONFIG.items():
            if key != 'gpu_id' and key != 'allow_large_jvms' and key != 'h2o_cluster_startup_timeout':
                init_args[key] = value
                
        if 'gpu_id' in H2O_CONFIG and H2O_CONFIG.get('backend') == 'gpu':
            init_args['gpu_id'] = H2O_CONFIG['gpu_id']
            
        print(f"Initializing H2O with settings from config: {init_args}")
        h2o.init(**init_args)
    
    # Print H2O info
    h2o.cluster().show_status()
    
    return h2o


def init_h2o_sparkling_water(spark, dynamic_config=True):
    """
    Initialize H2O Sparkling Water with GPU and high memory support.
    
    Args:
        spark (SparkSession): The Spark session
        dynamic_config (bool): Whether to dynamically adjust configuration
        
    Returns:
        H2OContext: Initialized H2O context
    """
    # Enable GPU support via environment variables if available
    resources = get_system_resources() if dynamic_config else None
    
    if dynamic_config and resources['gpu_count'] > 0:
        # Set environment variables for GPU support in Sparkling Water
        os.environ["H2O_DRIVER_USE_GPU"] = "true"
        os.environ["H2O_DISABLE_STRICT_VERSION_CHECK"] = "true"
    
    # Initialize H2O Sparkling Water context with appropriate settings
    try:
        # Initialize with extended settings
        h2o_context = H2OContext.getOrCreate(
            spark,
            extraProperties={
                "spark.ext.h2o.nthreads": "-1",  # Use all threads
                "spark.ext.h2o.cluster.size": "1",  # Single node mode
                "spark.ext.h2o.allow.large.jvms": "true",  # Allow large memory sizes
                "spark.ext.h2o.cluster.startup.timeout": "600",  # Longer timeout
            }
        )
        
        # Log H2O Sparkling Water information
        print(f"H2O Sparkling Water version: {h2o_context.getSparklingWaterVersion()}")
        print(f"H2O Flow UI: {h2o_context.flowURL()}")
        
    except Exception as e:
        print(f"Warning: Failed to initialize H2O Sparkling Water: {e}")
        print("Falling back to standalone H2O mode...")
        # Initialize standalone H2O as a fallback
        h2o.init()
        h2o_context = None
    
    return h2o_context


def get_spark_h2o_environment(dynamic_config=True, use_gpu=None):
    """
    Set up the complete Spark and H2O environment optimized for high memory and GPU.
    
    Args:
        dynamic_config (bool): Whether to dynamically adjust configuration
        use_gpu (bool): Whether to explicitly use GPUs. If None, auto-detected
        
    Returns:
        tuple: (spark, h2o, h2o_context)
    """
    # Check for GPU availability if needed
    if use_gpu is None and dynamic_config:
        resources = get_system_resources()
        use_gpu = resources['gpu_count'] > 0
    
    # Configure environment for GPUs if available
    if use_gpu:
        os.environ["JAVA_OPTS"] = "-Dai.h2o.ext.backend.gpu.enable=true"
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # Make all GPUs visible by default
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(HARDWARE_CONFIG.get('gpu_count', 3))))
    
    # Create Spark session
    spark = create_spark_session(dynamic_config)
    
    # Initialize H2O
    h2o_instance = init_h2o(dynamic_config)
    
    # Initialize H2O Sparkling Water
    h2o_context = init_h2o_sparkling_water(spark, dynamic_config)
    
    # Print environment information
    print(f"Spark version: {spark.version}")
    print(f"H2O version: {h2o.version()}")
    print(f"GPU enabled: {use_gpu}")
    
    # Print available memory
    total_mem = psutil.virtual_memory().total / (1024**3)
    avail_mem = psutil.virtual_memory().available / (1024**3)
    print(f"System memory: {total_mem:.1f}GB total, {avail_mem:.1f}GB available")
    
    # Log GPU information if available
    if use_gpu:
        try:
            # Try to get GPU information using nvidia-smi via shell
            import subprocess
            gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode()
            print("GPU Information:")
            for line in gpu_info.split('\n'):
                if "MiB" in line:  # Print memory usage lines
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"Could not get detailed GPU information: {e}")
    
    return spark, h2o_instance, h2o_context


def select_gpu(gpu_id=0):
    """
    Select a specific GPU for use with models.
    
    Args:
        gpu_id (int): The GPU ID to use (0-indexed)
        
    Returns:
        dict: Updated configuration with the selected GPU
    """
    if gpu_id >= HARDWARE_CONFIG.get('gpu_count', 0):
        raise ValueError(f"GPU ID {gpu_id} is not available. Only {HARDWARE_CONFIG.get('gpu_count', 0)} GPUs detected.")
    
    # Set environment variable for CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create updated XGBoost parameters with this GPU
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params['gpu_id'] = gpu_id
    
    print(f"Selected GPU {gpu_id} for modeling")
    return xgb_params