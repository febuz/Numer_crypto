# Utils package

def get_h2o_init_params(**kwargs):
    """
    Get standardized H2O initialization parameters with correct temp directory.
    
    Args:
        **kwargs: Additional parameters to override defaults
        
    Returns:
        dict: H2O initialization parameters
    """
    # Ensure H2O temp directory exists
    import os
    os.makedirs("/media/knight2/EDB/tmp/h2o", mode=0o755, exist_ok=True)
    
    # Set environment variables for Java
    os.environ['TMPDIR'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['TMP'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['TEMP'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['JAVA_OPTS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
    os.environ['_JAVA_OPTIONS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
    
    default_params = {
        "ice_root": "/media/knight2/EDB/tmp/h2o",
        "verbose": False,
        "jvm_custom_args": ["-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o"]
    }
    default_params.update(kwargs)
    return default_params