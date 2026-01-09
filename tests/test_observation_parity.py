
import sys
import os
import numpy as np

# Add src to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.core.builders.observation_builder_v19 import ObservationBuilderV19
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for direct execution if path setup fails
    sys.path.append(os.path.join(project_root, 'src'))
    try:
        from core.builders.observation_builder_v19 import ObservationBuilderV19
    except ImportError:
        print("CRITICAL: Include to import ObservationBuilderV19. Check paths.")
        sys.exit(1)

def test_parity():
    print("="*60)
    print("V19 OBSERVATION PARITY TEST")
    print("="*60)
    
    try:
        print("Initializing ObservationBuilderV19...")
        builder = ObservationBuilderV19(config_path=os.path.join(project_root, 'config', 'feature_config_v19.json'))
        
        # Mock data (Standardized/Normalized values roughly)
        market_features = {
            "log_ret_5m": 0.001,
            "log_ret_1h": 0.002,
            "log_ret_4h": 0.005,
            "rsi_9": 55.0,
            "atr_pct": 0.0015,
            "adx_14": 25.0,
            "dxy_z": 0.5,
            "dxy_change_1d": 0.01,
            "vix_z": -0.2,
            "embi_z": 1.0,
            "brent_change_1d": -0.02,
            "rate_spread": 0.1,
            "usdmxn_change_1d": 0.005
        }
        
        position = 1.0
        time_norm = 0.5
        
        print(f"Building observation with {len(market_features)} core features + 2 state features...")
        obs = builder.build(market_features, position, time_norm)
        
        print(f"\nObservation Shape: {obs.shape}")
        print(f"Observation Dtype: {obs.dtype}")
        print(f"Observation Range: [{obs.min():.4f}, {obs.max():.4f}]")
        print("\nVector Content:")
        print(obs)
        
        expected_shape = (15,)
        if obs.shape == expected_shape:
            print(f"\n✅ PASS: Observation dimension matches V19 spec {expected_shape}.")
        else:
            print(f"\n❌ FAIL: Expected {expected_shape}, got {obs.shape}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_parity()
