#!/usr/bin/env python3
"""
Configuration Manager for HyperGraph Test UI

Handles saving, loading, and managing test configurations for the HyperGraph
deepfake detection system. Provides templates and validation for configurations.

Author: Quanty 7
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class ConfigManager:
    """Manages test configurations, templates, and validation."""
    
    def __init__(self, configs_dir: str = "web_ui/configs", templates_dir: str = "web_ui/config_templates"):
        self.configs_dir = Path(configs_dir)
        self.templates_dir = Path(templates_dir)
        
        # Create directories if they don't exist
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default templates
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default configuration templates."""
        uncertainty_defaults = {
            "uncertainty_head": "none",
            "mc_dropout_samples": 0,
            "batchensemble_members": 4,
            "sngp_hidden_dim": 256,
            "sngp_rff_dim": 256,
            "uncertainty_dropout_rate": 0.2,
            "uncertainty_train_frequency": 10,
            "graph_uncertainty_methods": "attribute_distance,embedding_distance,hybrid_distance",
            "graph_degree_penalty_weight": 1.0,
            "build_val_test_edges": True
        }

        templates = {
            "basic_training": {
                "name": "Basic Training",
                "description": "Standard training configuration with default settings",
                "config": {
                    "traversal_type": "comprehensive",
                    "enable_traversal_switching": False,
                    "architectures": "resnestdf",
                    "num_epochs": 20,
                    "batch_size": 32,
                    "bias_hop_period": 100,
                    "seed": 42,
                    "quality_threshold": 0.500,
                    "symmetry_threshold": 0.300,
                    "embedding_threshold": 0.700,
                    "cached_nodes": False,
                    "cache_nodes": False,
                    "cached_nodes_count": 1000,
                    "cache_file": "node_cache/cached_nodes.pkl",
                    "use_dynamic_cache": False,
                    "fair_train": False,
                    "fair_test": False,
                    "enable_ivalue_viz": True,
                    "viz_track_nodes": 50,
                    "viz_sample_size": 200,
                    "bias_loss_weight": 0.1,
                    "num_workers": 0,
                    **uncertainty_defaults
                }
            },
            "bias_analysis": {
                "name": "Bias Analysis",
                "description": "Configuration optimized for bias analysis with I-value visualization",
                "config": {
                    "traversal_type": "i-value",
                    "enable_traversal_switching": False,
                    "architectures": "resnestdf",
                    "num_epochs": 30,
                    "batch_size": 16,
                    "bias_hop_period": 50,
                    "seed": 42,
                    "quality_threshold": 0.500,
                    "symmetry_threshold": 0.300,
                    "embedding_threshold": 0.700,
                    "cached_nodes": True,
                    "cache_nodes": True,
                    "cached_nodes_count": 1000,
                    "cache_file": "node_cache/cached_nodes.pkl",
                    "use_dynamic_cache": False,
                    "fair_train": True,
                    "fair_test": True,
                    "enable_ivalue_viz": True,
                    "viz_track_nodes": 50,
                    "viz_sample_size": 200,
                    "bias_loss_weight": 0.2,
                    "num_workers": 0,
                    **uncertainty_defaults
                }
            },
            "traversal_switching": {
                "name": "Traversal Switching",
                "description": "Multi-traversal configuration with switching between methods",
                "config": {
                    "traversal_type": "comprehensive",
                    "enable_traversal_switching": True,
                    "traversal_sequence": "comprehensive,i-value,i-value-cluster-hop",
                    "switch_epochs": "10,20",
                    "architectures": "resnestdf",
                    "num_epochs": 35,
                    "batch_size": 32,
                    "bias_hop_period": 100,
                    "seed": 42,
                    "quality_threshold": 0.500,
                    "symmetry_threshold": 0.300,
                    "embedding_threshold": 0.700,
                    "cached_nodes": False,
                    "cache_nodes": True,
                    "cached_nodes_count": 1000,
                    "cache_file": "node_cache/cached_nodes.pkl",
                    "use_dynamic_cache": False,
                    "fair_train": False,
                    "fair_test": False,
                    "enable_ivalue_viz": True,
                    "viz_track_nodes": 75,
                    "viz_sample_size": 300,
                    "bias_loss_weight": 0.1,
                    "num_workers": 0,
                    "disconnected_switching": False,
                    **uncertainty_defaults
                }
            },
            "comparison_study": {
                "name": "Comparison Study",
                "description": "Test all traversal types for comparison",
                "config": {
                    "test_all_traversals": True,
                    "architectures": "resnestdf,effnetdf",
                    "num_epochs": 25,
                    "batch_size": 32,
                    "bias_hop_period": 75,
                    "seed": 42,
                    "quality_threshold": 0.500,
                    "symmetry_threshold": 0.300,
                    "embedding_threshold": 0.700,
                    "cached_nodes": True,
                    "cache_nodes": True,
                    "cached_nodes_count": 1000,
                    "cache_file": "node_cache/cached_nodes.pkl",
                    "use_dynamic_cache": False,
                    "fair_train": True,
                    "fair_test": True,
                    "enable_ivalue_viz": True,
                    "viz_track_nodes": 100,
                    "viz_sample_size": 500,
                    "bias_loss_weight": 0.15,
                    "num_workers": 0,
                    **uncertainty_defaults
                }
            }
        }
        
        # Save templates if they don't exist
        for template_name, template_data in templates.items():
            template_file = self.templates_dir / f"{template_name}.json"
            if not template_file.exists():
                self._save_file(template_file, template_data)
    
    def _save_file(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Save data to a file with error handling."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving file {filepath}: {e}")
            return False
    
    def _load_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load data from a file with error handling."""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None
    
    def save_configuration(self, name: str, config: Dict[str, Any]) -> bool:
        """Save a test configuration.
        
        Saves the complete configuration dictionary exactly as provided. All fields
        in the config dictionary are preserved without modification. The full config
        dictionary is stored and will be reused exactly as saved when running tests.
        
        Args:
            name: Configuration name
            config: Complete configuration dictionary with all fields
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Add metadata wrapper, preserving the full config dictionary
        config_data = {
            "name": name,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "config": config  # Full config dictionary - all fields preserved
        }
        
        config_file = self.configs_dir / f"{name}.json"
        return self._save_file(config_file, config_data)
    
    def load_configuration(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a test configuration.
        
        Returns the full saved configuration including metadata. The inner 'config'
        key contains the complete configuration dictionary with all saved fields.
        
        Args:
            name: Configuration name
            
        Returns:
            Dictionary with 'name', 'created', 'modified', and 'config' keys,
            where 'config' contains the full saved configuration dictionary.
            Returns None if configuration not found.
        """
        config_file = self.configs_dir / f"{name}.json"
        return self._load_file(config_file)
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all saved configurations."""
        configs = []
        for config_file in self.configs_dir.glob("*.json"):
            config_data = self._load_file(config_file)
            if config_data:
                configs.append({
                    "name": config_data.get("name", config_file.stem),
                    "created": config_data.get("created", "Unknown"),
                    "modified": config_data.get("modified", "Unknown"),
                    "description": config_data.get("description", "")
                })
        return sorted(configs, key=lambda x: x["modified"], reverse=True)
    
    def delete_configuration(self, name: str) -> bool:
        """Delete a configuration."""
        try:
            config_file = self.configs_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting configuration {name}: {e}")
            return False
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all configuration templates."""
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            template_data = self._load_file(template_file)
            if template_data:
                templates.append({
                    "name": template_data.get("name", template_file.stem),
                    "description": template_data.get("description", ""),
                    "template_id": template_file.stem
                })
        return sorted(templates, key=lambda x: x["name"])
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific template."""
        template_file = self.templates_dir / f"{template_name}.json"
        return self._load_file(template_file)
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration and return validation results."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = [
            "traversal_type", "num_epochs", "batch_size", "seed",
            "quality_threshold", "symmetry_threshold", "embedding_threshold"
        ]
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validation rules
        if "num_epochs" in config:
            if not isinstance(config["num_epochs"], int) or config["num_epochs"] <= 0:
                errors.append("num_epochs must be a positive integer")
        
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                errors.append("batch_size must be a positive integer")
        
        # Traversal validation
        if config.get("enable_traversal_switching", False):
            if "traversal_sequence" not in config:
                errors.append("traversal_sequence required when traversal switching is enabled")
            if "switch_epochs" not in config:
                errors.append("switch_epochs required when traversal switching is enabled")
            
            # Validate traversal sequence and switch epochs
            if "traversal_sequence" in config and "switch_epochs" in config:
                try:
                    traversals = [t.strip() for t in config["traversal_sequence"].split(',')]
                    epochs = [int(e.strip()) for e in config["switch_epochs"].split(',')]
                    
                    if len(epochs) != len(traversals) - 1:
                        errors.append("Number of switch epochs must be one less than traversal sequence length")
                    
                    valid_traversals = ["comprehensive", "random", "i-value", "i-value-cluster-hop"]
                    for traversal in traversals:
                        if traversal not in valid_traversals:
                            errors.append(f"Invalid traversal type: {traversal}")
                
                except ValueError:
                    errors.append("switch_epochs must be comma-separated integers")
        
        # Threshold validation
        for threshold_field in ["quality_threshold", "symmetry_threshold", "embedding_threshold"]:
            if threshold_field in config:
                value = config[threshold_field]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    errors.append(f"{threshold_field} must be a number between 0 and 1")
        
        # Architecture validation
        if "architectures" in config:
            valid_archs = ["resnestdf", "efficientnet", "resnet50", "vgg16"]
            archs = [a.strip() for a in config["architectures"].split(',')]
            for arch in archs:
                if arch not in valid_archs:
                    warnings.append(f"Architecture '{arch}' may not be supported")
        
        # Performance warnings
        if config.get("enable_ivalue_viz", False) and config.get("num_epochs", 0) > 50:
            warnings.append("I-value visualization with >50 epochs may generate very large files")
        
        if config.get("batch_size", 32) > 64:
            warnings.append("Large batch sizes may cause memory issues")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def export_configuration(self, name: str, format: str = "json") -> Optional[str]:
        """Export a configuration to a specific format."""
        config = self.load_configuration(name)
        if not config:
            return None
        
        try:
            if format.lower() == "yaml":
                return yaml.dump(config, default_flow_style=False)
            else:  # Default to JSON
                return json.dumps(config, indent=2)
        except Exception as e:
            print(f"Error exporting configuration {name}: {e}")
            return None
    
    def import_configuration(self, name: str, data: str, format: str = "json") -> bool:
        """Import a configuration from formatted data."""
        try:
            if format.lower() == "yaml":
                config_data = yaml.safe_load(data)
            else:  # Default to JSON
                config_data = json.loads(data)
            
            # Extract config if it's wrapped in metadata
            if "config" in config_data:
                config = config_data["config"]
            else:
                config = config_data
            
            return self.save_configuration(name, config)
        
        except Exception as e:
            print(f"Error importing configuration {name}: {e}")
            return False 
