import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

class VGGPartitioner:
    def __init__(self):
        print("🔧 Loading VGG16 model...")
        # Load pre-trained VGG16
        self.full_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        print(f"VGG16 loaded with {len(self.full_model.layers)} layers")
    
    def load_imagenet_samples(self):
        """Load sample ImageNet images for testing"""
        images = []
        
        # Try to load from local files first
        local_image_paths = [
            "sample_images/dog.jpg",
            "sample_images/cat.jpg", 
            "sample_images/car.jpg"
        ]
        
        # Create sample_images directory if it doesn't exist
        sample_dir = "sample_images"
        if not os.path.exists(sample_dir):
            print(f"Creating {sample_dir} directory...")
            os.makedirs(sample_dir)
            print(f"Place test images in the {sample_dir} folder!")
        
        for img_path in local_image_paths:
            try:
                if os.path.exists(img_path):
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    name = os.path.basename(img_path).split('.')[0]
                    images.append((img_array, name))
                    print(f"✅ {name} image loaded successfully")
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
        
        if not images:
            print("No sample images found. Using dummy data instead.")
            print("To use real images, add JPG files to the sample_images/ folder")
        
        return images
        
    def add_differential_privacy(self, data, epsilon=1.0):
        """Add Gaussian noise for differential privacy"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Calculate noise scale (simplified - in practice, need proper sensitivity analysis)
        sensitivity = 1.0  # should be calculated based on specific use case
        noise_scale = sensitivity / epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, data.shape)
        noisy_data = data + noise
        
        print(f"Applied DP with ε={epsilon}, noise_scale={noise_scale:.4f}")
        return noisy_data
    
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
        
        print(f"📱 Client model: {len(client_model.layers)} layers")
        print(f"☁️  Server model: {len(server_model.layers)} layers")
        
        return client_model, server_model
    
    def test_inference_with_privacy(self, cut_layer, epsilon=None, use_real_data=True):
        """Test inference with optional differential privacy"""
        print(f"\n🧪 Testing inference with cut at layer {cut_layer}")
        if epsilon:
            print(f"🔒Privacy level: ε={epsilon}")
        
        # Get test data
        if use_real_data:
            images = self.load_imagenet_samples()
            if not images:
                print("No images loaded, falling back to dummy data")
                test_image = np.random.random((1, 224, 224, 3))
                image_name = "dummy"
            else:
                test_image, image_name = images[0]  # Use first image
        else:
            test_image = np.random.random((1, 224, 224, 3))
            image_name = "dummy"
        
        print(f"Testing with {image_name} image, shape: {test_image.shape}")
        
        # Partition the model
        client_model, server_model = self.partition_at_layer(cut_layer)
        
        # Client-side processing (Nano simulation)
        start_time = time.time()
        intermediate_data = client_model.predict(test_image, verbose=0)
        client_time = time.time() - start_time
        
        print(f"Client processing time: {client_time:.4f} seconds")
        print(f"Intermediate data shape: {intermediate_data.shape}")
        
        # Apply differential privacy if specified
        if epsilon:
            private_data = self.add_differential_privacy(intermediate_data, epsilon)
            transmission_data = private_data
        else:
            transmission_data = intermediate_data
        
        print(f"Data to transmit: {transmission_data.size * 4 / 1024:.2f} KB")
        
        # Server-side processing
        start_time = time.time()
        final_output = server_model.predict(transmission_data, verbose=0)
        server_time = time.time() - start_time
        
        print(f"Server processing time: {server_time:.4f} seconds")
        print(f"Total time: {client_time + server_time:.4f} seconds")
        
        # Show predictions
        if use_real_data:
            predictions = decode_predictions(final_output, top=3)[0]
            print(f"Top predictions for {image_name}:")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                print(f"  {i+1}. {label}: {score:.4f}")
        else:
            predicted_class = np.argmax(final_output[0])
            confidence = np.max(final_output[0])
            print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
        
        return {
            'cut_layer': cut_layer,
            'epsilon': epsilon,
            'image_name': image_name,
            'client_time': client_time,
            'server_time': server_time,
            'total_time': client_time + server_time,
            'intermediate_size_kb': transmission_data.size * 4 / 1024,
            'intermediate_shape': intermediate_data.shape,
            'final_output': final_output
        }

def main():
    print("VGG16 Partitioning with Differential Privacy Demo")
    print("=" * 60)
    
    partitioner = VGGPartitioner()
    
    # Test parameters based on Effect-DNN and Apple DP practices
    test_layers = [3, 5, 8, 10]  # Common cut points from literature
    privacy_levels = [None, 8.0, 4.0, 1.0, 0.1]  # None = no privacy, others based on Apple's usage
    
    results = []
    
    # Test each cut point with different privacy levels
    for layer in test_layers:
        print(f"\n{'='*20} TESTING LAYER {layer} {'='*20}")
        
        for epsilon in privacy_levels:
            try:
                result = partitioner.test_inference_with_privacy(
                    cut_layer=layer, 
                    epsilon=epsilon, 
                    use_real_data=True
                )
                results.append(result)
                print("-" * 50)
            except Exception as e:
                print(f"Error at layer {layer}, ε={epsilon}: {e}")
                print("-" * 50)
    
    # Summary analysis
    print(f"\nPERFORMANCE SUMMARY")
    print("=" * 80)
    print("Layer | Privacy (ε) | Client Time | Server Time | Total Time | Data Size")
    print("-" * 80)
    
    for r in results:
        epsilon_str = f"{r['epsilon']:.1f}" if r['epsilon'] else "None"
        print(f"{r['cut_layer']:5d} | {epsilon_str:11s} | {r['client_time']:10.4f}s | {r['server_time']:10.4f}s | {r['total_time']:9.4f}s | {r['intermediate_size_kb']:7.2f}KB")
    
    # Privacy-Performance Analysis
    print(f"\n🔍 PRIVACY-PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Group by layer and show privacy impact
    for layer in test_layers:
        layer_results = [r for r in results if r['cut_layer'] == layer]
        if len(layer_results) > 1:
            baseline = next((r for r in layer_results if r['epsilon'] is None), None)
            if baseline:
                print(f"\nLayer {layer} - Privacy Impact on Performance:")
                for r in layer_results:
                    if r['epsilon'] is not None:
                        time_overhead = ((r['total_time'] - baseline['total_time']) / baseline['total_time']) * 100
                        print(f"  ε={r['epsilon']:4.1f}: {time_overhead:+5.1f}% time overhead")

if __name__ == "__main__":
    main()