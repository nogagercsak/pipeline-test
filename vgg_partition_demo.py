import tensorflow as tf
import numpy as np
import time

class VGGPartitioner:
    def __init__(self):
        print("Loading VGG16 model...")
        # Load pre-trained VGG16
        self.full_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        print(f"VGG16 loaded with {len(self.full_model.layers)} layers")
    
    def partition_at_layer(self, cut_layer):
        """Split VGG16 into two parts at the specified layer"""
        print(f"Partitioning VGG16 at layer {cut_layer}")
        
        # Client side (layers 0 to cut_layer)
        client_input = self.full_model.input
        client_output = self.full_model.layers[cut_layer].output
        client_model = tf.keras.Model(inputs=client_input, outputs=client_output)
        
        # Server side (layers cut_layer+1 to end)
        server_input_shape = client_output.shape[1:]
        server_input = tf.keras.Input(shape=server_input_shape)
        x = server_input
        
        for layer in self.full_model.layers[cut_layer + 1:]:
            x = layer(x)
        
        server_model = tf.keras.Model(inputs=server_input, outputs=x)
        
        print(f"Client model: {len(client_model.layers)} layers")
        print(f"Server model: {len(server_model.layers)} layers")
        
        return client_model, server_model
    
    def test_inference(self, cut_layer):
        """Test the partitioned inference with dummy data"""
        print(f"\nTesting inference with cut at layer {cut_layer}")
        
        # Dummy image data
        dummy_image = np.random.random((1, 224, 224, 3))
        print(f"Input image shape: {dummy_image.shape}")
        
        # Partition the model
        client_model, server_model = self.partition_at_layer(cut_layer)
        
        # Nano
        start_time = time.time()
        intermediate_data = client_model.predict(dummy_image, verbose=0)
        client_time = time.time() - start_time
        
        print(f"Client processing time: {client_time:.4f} seconds")
        print(f"Intermediate data shape: {intermediate_data.shape}")
        print(f"Data to transmit: {intermediate_data.size * 4 / 1024:.2f} KB")  # 4 bytes per float32
        
        # Server
        start_time = time.time()
        final_output = server_model.predict(intermediate_data, verbose=0)
        server_time = time.time() - start_time
        
        print(f"Server processing time: {server_time:.4f} seconds")
        print(f"🎯 Final output shape: {final_output.shape}")
        print(f"⏱️  Total time: {client_time + server_time:.4f} seconds")
        
        # Show top prediction
        predicted_class = np.argmax(final_output[0])
        confidence = np.max(final_output[0])
        print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
        
        return {
            'cut_layer': cut_layer,
            'client_time': client_time,
            'server_time': server_time,
            'total_time': client_time + server_time,
            'intermediate_size_kb': intermediate_data.size * 4 / 1024,
            'intermediate_shape': intermediate_data.shape
        }

def main():
    print("VGG16 Partitioning Demo")
    print("=" * 50)
    
    partitioner = VGGPartitioner()
    
    # Test different cut points
    test_layers = [5, 10, 15] 
    results = []
    
    for layer in test_layers:
        try:
            result = partitioner.test_inference(layer)
            results.append(result)
            print("-" * 50)
        except Exception as e:
            print(f"Error at layer {layer}: {e}")
            print("-" * 50)
    
    # Summary
    print("\nSUMMARY:")
    print("Layer | Client Time | Server Time | Total Time | Data Size")
    print("-" * 60)
    for r in results:
        print(f"{r['cut_layer']:5d} | {r['client_time']:10.4f}s | {r['server_time']:10.4f}s | {r['total_time']:9.4f}s | {r['intermediate_size_kb']:7.2f}KB")

if __name__ == "__main__":
    main()