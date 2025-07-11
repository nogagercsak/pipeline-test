import tensorflow as tf
import sys
import platform

def test_setup():
    print("Environment Check")
    print("=" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Test basic operations
    try:
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print("TensorFlow operations working")
    except Exception as e:
        print(f"TensorFlow test failed: {e}")
        return False
    
    # Test model loading
    try:
        model = tf.keras.applications.MobileNet(weights='imagenet')
        print("Model loading working")
        del model  # Free memory
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_setup()