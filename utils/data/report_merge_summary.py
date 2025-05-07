"""
Utility to report summary information about data merges.
"""
import os
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def report_data_summary(numerai_data, yiedl_data, merged_data, output_dir=None):
    """
    Generate a summary report of data preparation
    
    Args:
        numerai_data: Dictionary with loaded Numerai data frames
        yiedl_data: Dictionary with loaded Yiedl data frames
        merged_data: Dictionary with merged datasets
        output_dir: Directory to save report (optional)
        
    Returns:
        dict: Summary information
    """
    # Create summary dictionary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'numerai_data': {},
        'yiedl_data': {},
        'merged_data': {},
        'symbols': {}
    }
    
    # Numerai data summary
    if 'current_round' in numerai_data:
        summary['numerai_data']['current_round'] = numerai_data['current_round']
    
    if 'train_targets' in numerai_data:
        summary['numerai_data']['train_targets_shape'] = numerai_data['train_targets'].shape
    
    if 'live_universe' in numerai_data:
        summary['numerai_data']['live_universe_shape'] = numerai_data['live_universe'].shape
    
    if 'train_data' in numerai_data:
        summary['numerai_data']['train_data_shape'] = numerai_data['train_data'].shape
    
    if 'symbols' in numerai_data:
        summary['numerai_data']['symbols_count'] = len(numerai_data['symbols'])
        summary['symbols']['numerai'] = numerai_data['symbols']
    
    # Yiedl data summary
    if 'latest' in yiedl_data:
        summary['yiedl_data']['latest_shape'] = yiedl_data['latest'].shape
    
    if 'historical' in yiedl_data:
        summary['yiedl_data']['historical_shape'] = yiedl_data['historical'].shape
    
    if 'symbols' in yiedl_data:
        summary['yiedl_data']['symbols_count'] = len(yiedl_data['symbols'])
        summary['symbols']['yiedl'] = yiedl_data['symbols']
    
    # Merged data summary
    if 'train' in merged_data:
        summary['merged_data']['train_shape'] = merged_data['train'].shape
    
    if 'live' in merged_data:
        summary['merged_data']['live_shape'] = merged_data['live'].shape
    
    if 'overlapping_symbols' in merged_data:
        summary['symbols']['overlapping'] = merged_data['overlapping_symbols']
        summary['symbols']['overlapping_count'] = len(merged_data['overlapping_symbols'])
    
    # Output locations
    if 'train_file' in merged_data:
        summary['merged_data']['train_file'] = merged_data['train_file']
    
    if 'live_file' in merged_data:
        summary['merged_data']['live_file'] = merged_data['live_file']
    
    # Print summary to log
    _print_summary(summary)
    
    # Save summary to file if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"data_summary_{date_str}.json")
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Data summary saved to {report_file}")
        summary['report_file'] = report_file
    
    return summary

def _print_summary(summary):
    """Print a formatted summary to the log"""
    logger.info("\n===== DATA PREPARATION SUMMARY =====")
    
    # Numerai information
    if 'numerai_data' in summary:
        logger.info("Numerai Data:")
        if 'current_round' in summary['numerai_data']:
            logger.info(f"  Current round: {summary['numerai_data']['current_round']}")
        
        for key, value in summary['numerai_data'].items():
            if key != 'current_round':
                logger.info(f"  {key}: {value}")
    
    # Yiedl information
    if 'yiedl_data' in summary:
        logger.info("\nYiedl Data:")
        for key, value in summary['yiedl_data'].items():
            logger.info(f"  {key}: {value}")
    
    # Symbol information
    if 'symbols' in summary:
        logger.info("\nSymbol Coverage:")
        
        if 'numerai' in summary['symbols']:
            num_symbols = len(summary['symbols']['numerai'])
            preview = summary['symbols']['numerai'][:5] if num_symbols > 5 else summary['symbols']['numerai']
            logger.info(f"  Numerai symbols: {num_symbols} (e.g., {', '.join(preview)}...)")
        
        if 'yiedl' in summary['symbols']:
            num_symbols = len(summary['symbols']['yiedl'])
            preview = summary['symbols']['yiedl'][:5] if num_symbols > 5 else summary['symbols']['yiedl']
            logger.info(f"  Yiedl symbols: {num_symbols} (e.g., {', '.join(preview)}...)")
        
        if 'overlapping' in summary['symbols']:
            overlap_symbols = summary['symbols']['overlapping']
            num_overlap = len(overlap_symbols)
            preview = overlap_symbols[:10] if num_overlap > 10 else overlap_symbols
            logger.info(f"  Overlapping symbols: {num_overlap} (e.g., {', '.join(preview)}...)")
    
    # Merged dataset information
    if 'merged_data' in summary:
        logger.info("\nMerged Datasets:")
        for key, value in summary['merged_data'].items():
            if 'file' in key:
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example usage
    # This would typically be called after creating merged datasets
    # See create_merged_dataset.py for an example