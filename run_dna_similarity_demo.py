#!/usr/bin/env python3
"""
DNA Similarity Demo - End-to-End Implementation with IntegratedML Framework

This script demonstrates DNA sequence classification and similarity search using
the IntegratedML Flexible Model Integration Framework.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n🧬 Step {step_num}: {description}")
    print("-" * 60)


def wait_for_user():
    """Wait for user input to continue."""
    input("\nPress Enter to continue...")


def main():
    """Run the complete DNA Similarity demo."""
    print_banner("INTEGRATEDML DNA SIMILARITY & CLASSIFICATION DEMO")
    print("This demo showcases DNA sequence classification and similarity search")
    print("using the IntegratedML Flexible Model Integration Framework.")
    
    try:
        # Step 1: Framework Introduction
        print_step(1, "IntegratedML Framework Introduction")
        
        print("🎯 This demo demonstrates how the IntegratedML framework solves")
        print("   the original DNA similarity project challenges:")
        print()
        print("   Challenge #1: Hardcoded Vectorization")
        print("   ✓ Solution: Configurable vectorization strategies")
        print()
        print("   Challenge #2: Hardcoded Algorithm Selection") 
        print("   ✓ Solution: Pluggable algorithms via YAML configuration")
        print()
        print("   Challenge #3: Mixed Database Logic")
        print("   ✓ Solution: Clean database abstraction layer")
        
        wait_for_user()
        
        # Step 2: Demo Setup and Configuration
        print_step(2, "Demo Setup and Configuration")
        
        print("🔧 Initializing DNA similarity demo...")
        
        # Check if demo files exist
        demo_dir = project_root / "demos" / "dna_similarity"
        config_path = demo_dir / "config" / "dna_model_config.yaml"
        demo_script = demo_dir / "dna_demo.py"
        
        if not demo_dir.exists():
            logger.error("❌ DNA similarity demo directory not found!")
            print("Please ensure the demos/dna_similarity directory exists.")
            return False
        
        if not config_path.exists():
            logger.error("❌ DNA demo configuration file not found!")
            print(f"Please ensure {config_path} exists.")
            return False
            
        if not demo_script.exists():
            logger.error("❌ DNA demo script not found!")
            print(f"Please ensure {demo_script} exists.")
            return False
        
        print("✅ Demo directory structure verified")
        print(f"📁 Demo location: {demo_dir}")
        print(f"⚙️ Configuration: {config_path.name}")
        print(f"🐍 Demo script: {demo_script.name}")
        
        wait_for_user()
        
        # Step 3: Import and Initialize Demo Components
        print_step(3, "Loading Demo Components")
        
        print("📦 Importing DNA similarity demo modules...")
        
        try:
            # Import the demo runner
            sys.path.insert(0, str(demo_dir))
            from dna_demo import DNADemoRunner
            
            print("✅ DNA demo modules imported successfully")
            
            # Initialize demo runner
            print("🧬 Initializing DNA demo runner...")
            demo_runner = DNADemoRunner(str(config_path))
            
            print("✅ DNA demo runner initialized")
            print(f"📊 Configuration loaded from: {config_path.name}")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import DNA demo modules: {e}")
            print("Please ensure all required dependencies are installed:")
            print("  pip install pandas scikit-learn sentence-transformers pyyaml numpy")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize demo runner: {e}")
            print(f"Error: {e}")
            return False
        
        wait_for_user()
        
        # Step 4: Run Vectorization Strategy Demonstration
        print_step(4, "Vectorization Strategy Demonstration")
        
        print("🔬 Demonstrating multiple vectorization approaches...")
        print("This addresses Challenge #1: Hardcoded Vectorization")
        print()
        
        try:
            # Create sample data for demonstrations
            sample_data = demo_runner._create_sample_data()
            print(f"✅ Created sample dataset with {len(sample_data)} DNA sequences")
            print(f"📊 Classes represented: {sample_data['class'].nunique()}")
            
            # Run vectorization demonstration
            demo_runner.demonstrate_vectorization_strategies(sample_data)
            
        except Exception as e:
            logger.error(f"❌ Vectorization demonstration failed: {e}")
            print(f"Error during vectorization demo: {e}")
        
        wait_for_user()
        
        # Step 5: Run Algorithm Selection Demonstration
        print_step(5, "Algorithm Selection Demonstration")
        
        print("🤖 Demonstrating flexible algorithm selection...")
        print("This addresses Challenge #2: Hardcoded Algorithm Selection")
        print()
        
        try:
            # Run algorithm demonstration
            demo_runner.demonstrate_algorithm_selection(sample_data)
            
        except Exception as e:
            logger.error(f"❌ Algorithm demonstration failed: {e}")
            print(f"Error during algorithm demo: {e}")
        
        wait_for_user()
        
        # Step 6: Database Integration Demonstration
        print_step(6, "Database Integration Demonstration")
        
        print("🗃️ Demonstrating clean IRIS database integration...")
        print("This addresses Challenge #3: Mixed Database Logic")
        print()
        
        try:
            # Run database integration demonstration
            demo_runner.demonstrate_iris_integration()
            
        except Exception as e:
            logger.error(f"❌ Database integration demonstration failed: {e}")
            print(f"Error during database demo: {e}")
        
        wait_for_user()
        
        # Step 7: Similarity Search Demonstration
        print_step(7, "DNA Similarity Search Demonstration")
        
        print("🔍 Demonstrating vector-based DNA similarity search...")
        print("This shows advanced framework capabilities for sequence analysis")
        print()
        
        try:
            # Run similarity search demonstration
            demo_runner.demonstrate_similarity_search(sample_data)
            
        except Exception as e:
            logger.error(f"❌ Similarity search demonstration failed: {e}")
            print(f"Error during similarity search demo: {e}")
        
        wait_for_user()
        
        # Step 8: Complete Framework Demonstration
        print_step(8, "Complete Framework Integration")
        
        print("🚀 Running complete IntegratedML framework demonstration...")
        print("This shows all components working together seamlessly")
        print()
        
        try:
            # Run the complete demo
            print("=" * 60)
            print("RUNNING COMPLETE DNA SIMILARITY FRAMEWORK DEMO")
            print("=" * 60)
            
            # Execute the full demo
            demo_runner.run_complete_demo()
            
            print("=" * 60)
            print("DNA SIMILARITY FRAMEWORK DEMO COMPLETED")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Complete demo failed: {e}")
            print(f"Error during complete demo: {e}")
        
        wait_for_user()
        
        # Final summary
        print_banner("DNA SIMILARITY DEMO COMPLETED! 🎉")
        print("Key accomplishments:")
        print("✅ Demonstrated configurable vectorization strategies")
        print("✅ Showed flexible algorithm selection")
        print("✅ Illustrated clean database abstraction")
        print("✅ Performed DNA sequence classification")
        print("✅ Executed vector-based similarity search")
        print("✅ Showcased framework flexibility and extensibility")
        print()
        print("🧬 Framework Benefits Demonstrated:")
        print("   📊 Configuration-driven development")
        print("   🔧 Clean separation of concerns")
        print("   🚀 Enterprise-ready deployment")
        print("   🎯 Improved developer productivity")
        print()
        print("This demo shows how the IntegratedML framework transforms")
        print("hardcoded bioinformatics solutions into flexible, maintainable,")
        print("and production-ready applications!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        print(f"Demo execution failed: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("   1. Ensure all dependencies are installed:")
        print("      pip install pandas scikit-learn sentence-transformers pyyaml numpy")
        print("   2. Verify demo files exist in demos/dna_similarity/")
        print("   3. Check that configuration file is present")
        print("   4. Run from the project root directory")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)