import tensorflow as tf
import numpy as np
import time
import os
import subprocess
import threading
from PIL import Image
from collections import defaultdict
import json
import glob

class EnergyMonitor:
    """Fixed Energy monitoring class"""
    
    def __init__(self):
        self.monitoring = False
        self.power_readings = []
        self.time_readings = []
        self.monitor_thread = None
        self.monitoring_method = "unknown"
        
        # Test if power monitoring is available
        self.power_available = self._test_power_monitoring()
        
    def _test_power_monitoring(self):
        """Test if power monitoring hardware is available"""
        # Try jtop first (most reliable for Jetson)
        try:
            print("🔍 Testing jtop (jetson-stats) power monitoring...")
            from jtop import jtop
            
            # Test if jtop can actually connect
            with jtop() as jetson:
                if jetson.ok():
                    print("✅ jtop (jetson-stats) found and working")
                    self.monitoring_method = "jtop"
                    return True
                else:
                    print("❌ jtop found but cannot connect to jetson")
        except ImportError:
            print("❌ jtop not installed")
        except Exception as e:
            print(f"❌ jtop test failed: {e}")
        
        # Try direct INA3221 power monitor paths for Jetson Nano
        jetson_power_paths = [
            # Most common Jetson Nano paths
            "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input", 
            "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power0_input",
            # Alternative INA3221 paths
            "/sys/bus/i2c/drivers/ina3221x/0-0040/iio_device/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input",
        ]
        
        print("🔍 Searching for Jetson Nano INA3221 power monitoring interfaces...")
        
        for path in jetson_power_paths:
            try:
                result = subprocess.check_output(f"cat {path}", shell=True, stderr=subprocess.DEVNULL)
                power_value = int(result.strip())
                print(f"✅ Power monitoring found at: {path}")
                print(f"   Current reading: {power_value} mW ({power_value/1000:.3f} W)")
                self.power_path = path
                self.monitoring_method = "ina3221"
                return True
            except subprocess.CalledProcessError:
                continue
            except Exception as e:
                continue
        
        # Try to find any working power monitoring
        print("🔍 Searching for any working power monitors...")
        try:
            # Find all power input files
            result = subprocess.check_output("find /sys -name 'in_power*_input' 2>/dev/null || true", shell=True)
            if result:
                print("Found power input files:")
                for path in result.decode().strip().split('\n'):
                    if path.strip():
                        try:
                            test_result = subprocess.check_output(f"cat {path.strip()}", shell=True, stderr=subprocess.DEVNULL)
                            power_value = int(test_result.strip())
                            print(f"✅ Working: {path.strip()} -> {power_value} mW")
                            self.power_path = path.strip()
                            self.monitoring_method = "direct"
                            return True
                        except:
                            print(f"❌ Non-working: {path.strip()}")
        except:
            pass
        
        print("⚠️  No hardware power monitoring found")
        return False
    
    def _monitor_power(self):
        """Background thread to monitor power consumption - FIXED for Jetson Nano"""
        start_time = time.time()
        
        while self.monitoring:
            power_mw = 7000  # Initialize with default value
            
            try:
                if self.power_available and self.monitoring_method == "jtop":
                    try:
                        from jtop import jtop
                        with jtop() as jetson:
                            if jetson.ok():
                                stats = jetson.stats
                                
                                # Method 1: Look for 'Power TOT' key (Jetson Nano format)
                                if 'Power TOT' in stats:
                                    power_mw = int(stats['Power TOT'])
                                    
                                # Method 2: Try jetson.power object (more detailed)
                                elif hasattr(jetson, 'power') and jetson.power:
                                    if 'tot' in jetson.power and 'power' in jetson.power['tot']:
                                        power_mw = int(jetson.power['tot']['power'])
                                    elif 'tot' in jetson.power and 'avg' in jetson.power['tot']:
                                        power_mw = int(jetson.power['tot']['avg'])
                                        
                                # Method 3: Sum individual CPU+GPU power rails
                                elif 'Power POM_5V_CPU' in stats and 'Power POM_5V_GPU' in stats:
                                    cpu_power = int(stats['Power POM_5V_CPU'])
                                    gpu_power = int(stats['Power POM_5V_GPU'])
                                    power_mw = cpu_power + gpu_power
                                    
                                # Method 4: Look for any power-related keys
                                else:
                                    total_power = 0
                                    power_keys_found = []
                                    for key in stats.keys():
                                        if 'Power' in key and isinstance(stats[key], (int, float)):
                                            total_power += stats[key]
                                            power_keys_found.append(f"{key}: {stats[key]}")
                                    
                                    if total_power > 0:
                                        power_mw = int(total_power)
                                        if len(power_keys_found) <= 3:  # Print debug info occasionally
                                            print(f"DEBUG: Using power keys: {', '.join(power_keys_found)}")
                                
                                # Sanity check - ensure reasonable power range for Jetson Nano
                                if 500 <= power_mw <= 25000:  # 0.5W to 25W range
                                    pass  # Keep the value
                                else:
                                    if power_mw != 7000:  # Don't warn about our default
                                        print(f"DEBUG: Unusual power reading: {power_mw} mW, using default")
                                    power_mw = 7000
                                    
                            else:
                                print("DEBUG: jtop not ready")
                                power_mw = 7000
                                
                    except Exception as e:
                        print(f"jtop power reading error: {e}")
                        power_mw = 7000
                        
                elif self.power_available and hasattr(self, 'power_path'):
                    # Use direct file reading as fallback
                    try:
                        power_output = subprocess.check_output(f"cat {self.power_path}", shell=True, stderr=subprocess.DEVNULL)
                        power_value = int(power_output.strip())
                        
                        # Auto-detect units based on value range
                        if power_value > 50000:  # Likely µW, convert to mW
                            power_mw = power_value // 1000
                        elif power_value > 50:  # Likely mW
                            power_mw = power_value
                        else:  # Likely W, convert to mW
                            power_mw = power_value * 1000
                    except Exception as e:
                        print(f"Direct power reading error: {e}")
                        power_mw = 7000
                        
                current_time = time.time() - start_time
                self.power_readings.append(power_mw)
                self.time_readings.append(current_time)
                
                time.sleep(0.2)  # Sample every 200ms
                
            except Exception as e:
                print(f"Power monitoring error: {e}")
                current_time = time.time() - start_time
                self.power_readings.append(power_mw)
                self.time_readings.append(current_time)
                time.sleep(0.2)
        
    def start_monitoring(self):
        """Start energy monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.power_readings = []
        self.time_readings = []
        self.monitor_thread = threading.Thread(target=self._monitor_power, daemon=True)
        self.monitor_thread.start()
        print("🔋 Energy monitoring started")
    
    def stop_monitoring(self):
        """Stop energy monitoring and return metrics"""
        if not self.monitoring:
            return {}
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.power_readings:
            return {}
        
        # Calculate energy metrics
        avg_power_mw = np.mean(self.power_readings)
        max_power_mw = np.max(self.power_readings)
        min_power_mw = np.min(self.power_readings)
        
        # Calculate energy consumption (integrate power over time)
        if len(self.time_readings) > 1:
            total_time_s = self.time_readings[-1] - self.time_readings[0]
            # Simple integration: average power × time
            energy_mj = (avg_power_mw / 1000.0) * total_time_s  # millijoules
        else:
            total_time_s = 0
            energy_mj = 0
        
        metrics = {
            'avg_power_mw': avg_power_mw,
            'max_power_mw': max_power_mw,
            'min_power_mw': min_power_mw,
            'total_time_s': total_time_s,
            'energy_mj': energy_mj,
            'energy_j': energy_mj / 1000.0,
            'samples_collected': len(self.power_readings),
            'power_available': self.power_available
        }
        
        print(f"🔋 Energy monitoring stopped - {len(self.power_readings)} samples collected")
        return metrics


class VGGPartitioner:
    def __init__(self):
        print("🔧 Loading VGG16 model...")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Initialize energy monitor
        self.energy_monitor = EnergyMonitor()
        
        # Check TensorFlow version and adjust imports accordingly
        tf_major = int(tf.__version__.split('.')[0])
        tf_minor = int(tf.__version__.split('.')[1])
        
        if tf_major == 1 or (tf_major == 2 and tf_minor < 2):
            # Older TensorFlow versions
            try:
                from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
                self.keras = tf.keras
                self.preprocess_input = preprocess_input
                self.decode_predictions = decode_predictions
            except ImportError:
                print("Error: Could not import VGG16. Please install keras-preprocessing:")
                print("pip3 install keras-preprocessing")
                raise
        else:
            # Newer TensorFlow versions
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
            self.keras = tf.keras
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
        
        # Load pre-trained VGG16
        self.full_model = VGG16(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        print(f"VGG16 loaded with {len(self.full_model.layers)} layers")
    
    def load_imagenet_samples(self, num_samples=5, dataset_path=None):
        """Load sample ImageNet images for testing"""
        images = []
        
        # Option 1: Load from specified dataset path
        if dataset_path and os.path.exists(dataset_path):
            print(f"🔍 Loading images from dataset: {dataset_path}")
            
            # Support different dataset structures
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            all_images = []
            
            for ext in image_extensions:
                # Search recursively for images
                pattern = os.path.join(dataset_path, '**', ext)
                found_images = glob.glob(pattern, recursive=True)
                all_images.extend(found_images)
            
            if all_images:
                # Randomly sample if we have more images than needed
                import random
                selected_images = random.sample(all_images, min(len(all_images), num_samples))
                
                for img_path in selected_images:
                    try:
                        img = Image.open(img_path)
                        img = img.resize((224, 224))
                        img = img.convert('RGB')
                        img_array = np.array(img, dtype=np.float32)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = self.preprocess_input(img_array)
                        name = os.path.basename(img_path).split('.')[0]
                        images.append((img_array, name))
                        print(f"✅ {name} loaded from {os.path.basename(os.path.dirname(img_path))}")
                    except Exception as e:
                        print(f"Failed to load {img_path}: {e}")
                        continue
                
                if images:
                    print(f"✅ Loaded {len(images)} real images from dataset")
                    return images
        
        # Option 2: Try to download COCO validation images (small sample)
        print("🔍 Attempting to download sample COCO images...")
        coco_sample_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
            "http://images.cocodataset.org/val2017/000000374628.jpg",  # person
            "http://images.cocodataset.org/val2017/000000252219.jpg",  # car
            "http://images.cocodataset.org/val2017/000000397133.jpg",  # dog
            "http://images.cocodataset.org/val2017/000000037777.jpg"   # food
        ]
        
        sample_dir = "sample_images"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        try:
            import urllib.request
            for i, url in enumerate(coco_sample_urls[:num_samples]):
                filename = f"coco_sample_{i}.jpg"
                filepath = os.path.join(sample_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                
                try:
                    img = Image.open(filepath)
                    img = img.resize((224, 224))
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess_input(img_array)
                    images.append((img_array, filename.split('.')[0]))
                    print(f"✅ {filename} downloaded and loaded")
                except Exception as e:
                    print(f"Failed to process downloaded {filename}: {e}")
                    
            if images:
                print(f"✅ Downloaded and loaded {len(images)} COCO images")
                return images
                
        except ImportError:
            print("urllib not available for downloading")
        except Exception as e:
            print(f"Failed to download COCO images: {e}")
        
        # Option 3: Try to load from local sample_images directory
        print("🔍 Looking for local images in sample_images/ directory...")
        
        if os.path.exists(sample_dir):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            all_local_images = []
            
            for ext in image_extensions:
                pattern = os.path.join(sample_dir, ext)
                found_images = glob.glob(pattern)
                all_local_images.extend(found_images)
            
            for img_path in all_local_images[:num_samples]:
                try:
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess_input(img_array)
                    name = os.path.basename(img_path).split('.')[0]
                    images.append((img_array, name))
                    print(f"✅ {name} loaded from sample_images/")
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")
        
        if images:
            print(f"✅ Loaded {len(images)} local images")
            return images
        
        # Option 4: Create dummy data as fallback
        print("⚠️ No real images found. Using dummy data.")
        print("\n📥 TO USE REAL IMAGES:")
        print("Option 1: Put images in sample_images/ folder:")
        print("  mkdir sample_images")
        print("  # Add .jpg, .png, .bmp files to sample_images/")
        print("\nOption 2: Use existing dataset:")
        print("  partitioner.load_imagenet_samples(dataset_path='/path/to/your/dataset')")
        print("\nOption 3: Download ImageNet validation set:")
        print("  wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar")
        print("\nOption 4: Use COCO dataset:")
        print("  # Images will be auto-downloaded on first run")
        
        # Create dummy data
        for i in range(num_samples):
            test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_image = self.preprocess_input(test_image)
            images.append((test_image, f"dummy_{i}"))
        
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
        client_model = self.keras.Model(inputs=client_input, outputs=client_output)
        
        # Server side (layers cut_layer+1 to end)
        server_input_shape = client_output.shape[1:]
        server_input = self.keras.Input(shape=server_input_shape)
        x = server_input
        
        for layer in self.full_model.layers[cut_layer + 1:]:
            x = layer(x)
        
        server_model = self.keras.Model(inputs=server_input, outputs=x)
        
        print(f"📱 Client model: {len(client_model.layers)} layers")
        print(f"☁️  Server model: {len(server_model.layers)} layers")
        
        return client_model, server_model
    
    def test_inference_with_privacy_and_energy(self, cut_layer, epsilon=None, use_real_data=True, warmup_runs=2, dataset_path=None):
        """Test inference with optional differential privacy and energy monitoring"""
        print(f"\n🧪 Testing inference with cut at layer {cut_layer}")
        if epsilon:
            print(f"🔒Privacy level: ε={epsilon}")
        
        # Get test data
        if use_real_data:
            images = self.load_imagenet_samples(dataset_path=dataset_path)
            if not images:
                print("No images loaded, falling back to dummy data")
                test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
                test_image = self.preprocess_input(test_image)
                image_name = "dummy"
            else:
                test_image, image_name = images[0]  # Use first image
        else:
            test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_image = self.preprocess_input(test_image)
            image_name = "dummy"
        
        print(f"Testing with {image_name} image, shape: {test_image.shape}")
        
        # Partition the model
        client_model, server_model = self.partition_at_layer(cut_layer)
        
        # Warmup runs to stabilize performance (don't measure these)
        print(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = client_model.predict(test_image, verbose=0)
            _ = server_model.predict(client_model.predict(test_image, verbose=0), verbose=0)
        
        # Start energy monitoring for the actual measurement
        self.energy_monitor.start_monitoring()
        
        # Client-side processing with timing
        client_start_time = time.time()
        intermediate_data = client_model.predict(test_image, verbose=0)
        
        # Apply differential privacy if specified
        if epsilon:
            transmission_data = self.add_differential_privacy(intermediate_data, epsilon)
        else:
            transmission_data = intermediate_data
        
        client_end_time = time.time()
        client_time = client_end_time - client_start_time
        
        # Server-side processing with timing
        server_start_time = time.time()
        final_output = server_model.predict(transmission_data, verbose=0)
        server_end_time = time.time()
        server_time = server_end_time - server_start_time
        
        # Stop energy monitoring and get metrics
        energy_metrics = self.energy_monitor.stop_monitoring()
        
        total_time = client_time + server_time
        
        print(f"Client processing time: {client_time:.4f} seconds")
        print(f"Server processing time: {server_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Intermediate data shape: {intermediate_data.shape}")
        print(f"Data to transmit: {transmission_data.size * 4 / 1024:.2f} KB")
        
        # Print energy metrics
        if energy_metrics:
            print(f"🔋 Energy Metrics:")
            print(f"   Average Power: {energy_metrics['avg_power_mw']:.2f} mW ({energy_metrics['avg_power_mw']/1000:.3f} W)")
            print(f"   Peak Power: {energy_metrics['max_power_mw']:.2f} mW ({energy_metrics['max_power_mw']/1000:.3f} W)")
            print(f"   Total Energy: {energy_metrics['energy_mj']:.2f} mJ ({energy_metrics['energy_j']:.6f} J)")
            print(f"   Monitoring Duration: {energy_metrics['total_time_s']:.4f} s")
            print(f"   Power Samples: {energy_metrics['samples_collected']}")
            print(f"   Hardware Available: {energy_metrics['power_available']}")
        
        # Show predictions
        if use_real_data and hasattr(self, 'decode_predictions'):
            try:
                predictions = self.decode_predictions(final_output, top=3)[0]
                print(f"Top predictions for {image_name}:")
                for i, (imagenet_id, label, score) in enumerate(predictions):
                    print(f"  {i+1}. {label}: {score:.4f}")
            except:
                predicted_class = np.argmax(final_output[0])
                confidence = np.max(final_output[0])
                print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
        else:
            predicted_class = np.argmax(final_output[0])
            confidence = np.max(final_output[0])
            print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
        
        # Combine all results
        result = {
            'cut_layer': cut_layer,
            'epsilon': epsilon,
            'image_name': image_name,
            'client_time': client_time,
            'server_time': server_time,
            'total_time': total_time,
            'intermediate_size_kb': transmission_data.size * 4 / 1024,
            'intermediate_shape': intermediate_data.shape,
            'final_output': final_output,
            'energy_metrics': energy_metrics
        }
        
        return result

def main():
    print("VGG16 Partitioning with Differential Privacy and Energy Monitoring Demo")
    print("=" * 80)
    
    try:
        partitioner = VGGPartitioner()
    except Exception as e:
        print(f"Failed to initialize VGGPartitioner: {e}")
        print("\nTrying to install missing dependencies...")
        print("Run: pip3 install keras-preprocessing Pillow")
        return
    
    # Test parameters
    test_layers = [3, 5, 8, 10]  # Common cut points
    privacy_levels = [None, 8.0, 4.0, 1.0]  # Privacy levels
    
    results = []
    
    # Test each configuration
    for layer in test_layers:
        print(f"\n{'='*20} TESTING LAYER {layer} {'='*20}")
        
        for epsilon in privacy_levels:
            try:
                result = partitioner.test_inference_with_privacy_and_energy(
                    cut_layer=layer, 
                    epsilon=epsilon, 
                    use_real_data=True,
                    warmup_runs=2
                )
                results.append(result)
                print("-" * 60)
            except Exception as e:
                print(f"Error at layer {layer}, ε={epsilon}: {e}")
                import traceback
                traceback.print_exc()
                print("-" * 60)
    
    # Summary analysis
    if results:
        print(f"\nPERFORMANCE & ENERGY SUMMARY")
        print("=" * 100)
        print("Layer | Privacy (ε) | Client Time | Server Time | Total Time | Data Size | Avg Power | Energy")
        print("-" * 100)
        
        for r in results:
            epsilon_str = f"{r['epsilon']:.1f}" if r['epsilon'] else "None"
            energy_metrics = r.get('energy_metrics', {})
            avg_power_w = energy_metrics.get('avg_power_mw', 0) / 1000.0
            energy_mj = energy_metrics.get('energy_mj', 0)
            
            print(f"{r['cut_layer']:5d} | {epsilon_str:11s} | {r['client_time']:10.4f}s | {r['server_time']:10.4f}s | {r['total_time']:9.4f}s | {r['intermediate_size_kb']:7.2f}KB | {avg_power_w:8.3f}W | {energy_mj:7.2f}mJ")
        
        # Save detailed results to JSON
        results_for_json = []
        for r in results:
            result_copy = r.copy()
            # Convert numpy arrays to lists for JSON serialization
            if 'final_output' in result_copy:
                del result_copy['final_output']  # Too large for JSON
            if 'intermediate_shape' in result_copy:
                result_copy['intermediate_shape'] = list(result_copy['intermediate_shape'])
            
            # Convert numpy/int64 types to native Python types
            if 'energy_metrics' in result_copy and result_copy['energy_metrics']:
                for key, value in result_copy['energy_metrics'].items():
                    if hasattr(value, 'item'):  # numpy scalar
                        result_copy['energy_metrics'][key] = value.item()
                    elif isinstance(value, (np.integer, np.floating)):
                        result_copy['energy_metrics'][key] = value.item()
                    elif isinstance(value, np.ndarray):
                        result_copy['energy_metrics'][key] = value.tolist()
            
            # Convert other numpy types
            for key, value in result_copy.items():
                if hasattr(value, 'item'):  # numpy scalar
                    result_copy[key] = value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    result_copy[key] = value.item()
                elif isinstance(value, np.ndarray):
                    result_copy[key] = value.tolist()
            
            results_for_json.append(result_copy)
        
        with open('vgg_energy_results.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        print(f"\n💾 Detailed results saved to 'vgg_energy_results.json'")
        
        # Energy efficiency analysis
        print(f"\n⚡ ENERGY EFFICIENCY ANALYSIS")
        print("=" * 60)
        
        # Group by layer and show privacy vs energy trade-offs
        for layer in test_layers:
            layer_results = [r for r in results if r['cut_layer'] == layer]
            if len(layer_results) > 1:
                print(f"\nLayer {layer} - Privacy vs Energy Trade-offs:")
                baseline = next((r for r in layer_results if r['epsilon'] is None), None)
                if baseline and baseline.get('energy_metrics'):
                    baseline_energy = baseline['energy_metrics'].get('energy_mj', 0)
                    for r in layer_results:
                        if r['epsilon'] is not None and r.get('energy_metrics'):
                            current_energy = r['energy_metrics'].get('energy_mj', 0)
                            if baseline_energy > 0:
                                energy_overhead = ((current_energy - baseline_energy) / baseline_energy) * 100
                            else:
                                energy_overhead = 0
                            print(f"  ε={r['epsilon']:4.1f}: {energy_overhead:+5.1f}% energy overhead")
    else:
        print("No successful results to analyze.")

if __name__ == "__main__":
    main()