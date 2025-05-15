"""
Simple test script to verify that pruning mode names are working correctly.
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from alignment.config import ExtraConfig, ExtraArgs

def test_pruning_modes():
    """Test that both config classes accept the standard pruning mode names."""
    
    logger.info("Testing ExtraConfig with standardized pruning mode names:")
    
    # Test ExtraConfig
    for pruning_mode in ["global_joint", "layer_wise", "layer_isolated", "cascading_layer"]:
        config = ExtraConfig(dropout_pruning_mode=pruning_mode)
        try:
            config.validate()
            logger.info(f"  ✓ ExtraConfig validates pruning_mode={pruning_mode}")
        except ValueError as e:
            logger.error(f"  ✗ ExtraConfig failed with pruning_mode={pruning_mode}: {e}")
    
    logger.info("\nTesting ExtraArgs with standardized pruning mode names:")
    
    # Test ExtraArgs
    for pruning_mode in ["global_joint", "layer_wise", "layer_isolated", "cascading_layer"]:
        args = ExtraArgs(dropout_pruning_mode=pruning_mode)
        try:
            args.validate()
            logger.info(f"  ✓ ExtraArgs validates pruning_mode={pruning_mode}")
        except ValueError as e:
            logger.error(f"  ✗ ExtraArgs failed with pruning_mode={pruning_mode}: {e}")
    
    # Test if old names are still being accepted
    logger.info("\nTesting compatibility with old pruning mode names:")
    
    for old_name, new_name in [
        ("global", "global_joint"),
        ("per_layer_combined", "layer_wise"),
        ("per_layer_independent", "layer_isolated")
    ]:
        config = ExtraConfig(dropout_pruning_mode=old_name)
        args = ExtraArgs(dropout_pruning_mode=old_name)
        
        try:
            config.validate()
            logger.info(f"  ✓ ExtraConfig still accepts old name {old_name} (should be {new_name})")
        except ValueError as e:
            logger.error(f"  ✗ ExtraConfig rejects old name {old_name}: {e}")
            
        try:
            args.validate()
            logger.info(f"  ✓ ExtraArgs still accepts old name {old_name} (should be {new_name})")
        except ValueError as e:
            logger.error(f"  ✗ ExtraArgs rejects old name {old_name}: {e}")
    
    logger.info("\nTest complete.")

if __name__ == "__main__":
    test_pruning_modes() 