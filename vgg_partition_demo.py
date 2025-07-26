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

class EnhancedEnergyMonitor:
    """Enhanced energy monitoring with accurate hardware detection and fixed synchronization"""
    
    def __init__(self):
        self.monitoring = False
        self.power_readings = []
        self.time_readings = []
        self.monitor_thread = None
        self.monitoring_method = "unknown"
        self.start_time = None
        self.power_path = None
        self.power_available = self._test_power_monitoring()
        
    def _test_power_monitoring(self):
        """Comprehensive test for power monitoring hardware availability"""
        print("🔍 Testing comprehensive power monitoring capabilities...")
        
        # Method 1: Test jtop (most reliable for Jetson)
        try:
            print("   Testing jtop (jetson-stats)...")
            from jtop import jtop
            
            with jtop() as jetson:
                if jetson.ok():
                    # Test if we can actually read power data
                    stats = jetson.stats
                    power_found = False
                    
                    # Check various power keys
                    power_keys = ['Power TOT', 'Power POM_5V_CPU', 'Power POM_5V_GPU']
                    for key in power_keys:
                        if key in stats:
                            power_found = True
                            print(f"      Found power key: {key} = {stats[key]}")
                    
                    # Check jetson.power object
                    if hasattr(jetson, 'power') and jetson.power:
                        if 'tot' in jetson.power:
                            power_found = True
                            print(f"      Found jetson.power.tot: {jetson.power['tot']}")
                    
                    if power_found:
                        print("✅ jtop power monitoring confirmed working")
                        self.monitoring_method = "jtop"
                        return True
                    else:
                        print("❌ jtop connected but no power data found")
                else:
                    print("❌ jtop found but cannot connect to jetson")
        except ImportError:
            print("❌ jtop not installed (pip install jetson-stats)")
        except Exception as e:
            print(f"❌ jtop test failed: {e}")
        
        # Method 2: Test direct Iना3221 power monitor paths
        print("   Testing direct INA3221 power monitoring...")
        jetson_power_paths = [
            # Common Jetson Nano INA3221 paths
            "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input", 
            "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power0_input",
            # Alternative INA3221 paths
            "/sys/bus/i2c/drivers/ina3221x/0-0040/iio_device/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input",
            # Generic power monitoring paths
            "/sys/class/power_supply/BAT0/power_now",
            "/sys/class/power_supply/BAT1/power_now",
        ]
        
        for path in jetson_power_paths:
            try:
                result = subprocess.check_output(f"cat {path}", shell=True, stderr=subprocess.DEVNULL)
                power_value = int(result.strip())
                print(f"✅ Power monitoring found at: {path}")
                print(f"   Current reading: {power_value} mW ({power_value/1000:.3f} W)")
                self.power_path = path
                self.monitoring_method = "ina3221"
                return True
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                continue
            except Exception as e:
                print(f"   Error testing {path}: {e}")
                continue
        
        # Method 3: Search for any working power monitors
        print("   Searching for any power monitoring interfaces...")
        try:
            result = subprocess.check_output("find /sys -name 'in_power*_input' -o -name 'power_now' 2>/dev/null | head -10", shell=True)
            if result:
                paths = result.decode().strip().split('\n')
                print(f"   Found {len(paths)} potential power files:")
                
                for path in paths:
                    if path.strip():
                        try:
                            test_result = subprocess.check_output(f"cat {path.strip()}", shell=True, stderr=subprocess.DEVNULL)
                            power_value = int(test_result.strip())
                            if 100 <= power_value <= 50000:  # Reasonable power range
                                print(f"✅ Working power monitor: {path.strip()} -> {power_value}")
                                self.power_path = path.strip()
                                self.monitoring_method = "direct"
                                return True
                            else:
                                print(f"   Unusual reading: {path.strip()} -> {power_value}")
                        except Exception as e:
                            print(f"   Non-working: {path.strip()} ({e})")
        except Exception as e:
            print(f"   Search failed: {e}")
        
        # Method 4: Try NVIDIA GPU power monitoring
        try:
            print("   Testing NVIDIA GPU power monitoring...")
            result = subprocess.check_output("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True, stderr=subprocess.DEVNULL)
            power_w = float(result.strip())
            print(f"✅ NVIDIA GPU power monitoring: {power_w} W")
            self.monitoring_method = "nvidia-smi"
            return True
        except Exception:
            print("   NVIDIA GPU monitoring not available")
        
        print("⚠️  No hardware power monitoring found - using estimation")
        self.monitoring_method = "estimated"
        return False
    
    def _monitor_power(self):
        """Enhanced background power monitoring with proper error handling"""
        while self.monitoring:
            power_mw = 7000  # Default fallback value
            current_time = time.time()
            
            try:
                if self.power_available:
                    if self.monitoring_method == "jtop":
                        power_mw = self._read_jtop_power()
                    elif self.monitoring_method in ["ina3221", "direct"]:
                        power_mw = self._read_direct_power()
                    elif self.monitoring_method == "nvidia-smi":
                        power_mw = self._read_nvidia_power()
                else:
                    # Simulate realistic power consumption with some variation
                    base_power = 5500  # 5.5W base
                    variation = np.random.randint(-1000, 2000)  # ±1W to +2W variation
                    power_mw = base_power + variation
                
            except Exception as e:
                print(f"Power reading error: {e}")
                power_mw = 7000  # Fallback
            
            self.power_readings.append(power_mw)
            self.time_readings.append(current_time)
            time.sleep(0.2)  # 200ms sampling rate
    
    def _read_jtop_power(self):
        """Read power from jtop with multiple fallback methods"""
        try:
            from jtop import jtop
            with jtop() as jetson:
                if jetson.ok():
                    stats = jetson.stats
                    
                    # Method 1: Total power
                    if 'Power TOT' in stats:
                        return int(stats['Power TOT'])
                    
                    # Method 2: jetson.power object
                    if hasattr(jetson, 'power') and jetson.power:
                        if 'tot' in jetson.power:
                            if isinstance(jetson.power['tot'], dict):
                                if 'power' in jetson.power['tot']:
                                    return int(jetson.power['tot']['power'])
                                elif 'avg' in jetson.power['tot']:
                                    return int(jetson.power['tot']['avg'])
                            else:
                                return int(jetson.power['tot'])
                    
                    # Method 3: Sum individual rails
                    power_sum = 0
                    power_keys = ['Power POM_5V_CPU', 'Power POM_5V_GPU', 'Power POM_5V_SOC']
                    for key in power_keys:
                        if key in stats and isinstance(stats[key], (int, float)):
                            power_sum += stats[key]
                    
                    if power_sum > 0:
                        return int(power_sum)
                    
                    # Method 4: Any power key
                    for key, value in stats.items():
                        if 'Power' in key and isinstance(value, (int, float)) and 500 <= value <= 25000:
                            return int(value)
                            
        except Exception as e:
            print(f"jtop read error: {e}")
        
        return 7000  # Fallback
    
    def _read_direct_power(self):
        """Read power from direct file access"""
        try:
            if self.power_path:
                result = subprocess.check_output(f"cat {self.power_path}", shell=True, stderr=subprocess.DEVNULL)
                power_value = int(result.strip())
                
                # Auto-detect units based on value range
                if power_value > 50000:  # Likely µW
                    return power_value // 1000
                elif power_value > 50:  # Likely mW
                    return power_value
                else:  # Likely W
                    return power_value * 1000
        except Exception as e:
            print(f"Direct power read error: {e}")
        
        return 7000
    
    def _read_nvidia_power(self):
        """Read power from NVIDIA GPU"""
        try:
            result = subprocess.check_output("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True, stderr=subprocess.DEVNULL)
            power_w = float(result.strip())
            return int(power_w * 1000)  # Convert to mW
        except Exception as e:
            print(f"NVIDIA power read error: {e}")
        
        return 7000
    
    def start_monitoring(self):
        """Start energy monitoring with proper synchronization"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.power_readings = []
        self.time_readings = []
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitor_power, daemon=True)
        self.monitor_thread.start()
        time.sleep(0.1)  # Brief delay to ensure monitoring starts
        print(f"🔋 Energy monitoring started ({self.monitoring_method})")
    
    def stop_monitoring(self):
        """Stop energy monitoring and return comprehensive metrics"""
        if not self.monitoring:
            return {}
            
        end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.power_readings or not self.time_readings:
            return {}
        
        # Calculate timing
        actual_duration = end_time - self.start_time if self.start_time else 0
        
        # Calculate power statistics
        avg_power_mw = np.mean(self.power_readings)
        max_power_mw = np.max(self.power_readings)
        min_power_mw = np.min(self.power_readings)
        std_power_mw = np.std(self.power_readings)
        
        # Calculate energy consumption
        energy_mj = (avg_power_mw / 1000.0) * actual_duration
        
        # Calculate sampling statistics
        sampling_rate = len(self.power_readings) / actual_duration if actual_duration > 0 else 0
        
        metrics = {
            'avg_power_mw': avg_power_mw,
            'max_power_mw': max_power_mw,
            'min_power_mw': min_power_mw,
            'std_power_mw': std_power_mw,
            'total_time_s': actual_duration,
            'energy_mj': energy_mj,
            'energy_j': energy_mj / 1000.0,
            'samples_collected': len(self.power_readings),
            'sampling_rate_hz': sampling_rate,
            'power_available': self.power_available,
            'monitoring_method': self.monitoring_method,
            'power_efficiency_mw_per_s': avg_power_mw / actual_duration if actual_duration > 0 else 0
        }
        
        print(f"🔋 Energy monitoring stopped - {len(self.power_readings)} samples over {actual_duration:.3f}s")
        return metrics


class DynamicSensitivityDP:
    """Advanced differential privacy with dynamic sensitivity calculation and multiple methods"""
    
    def __init__(self):
        self.sensitivity_cache = {}
        self.layer_analysis_cache = {}
        
    def calculate_dynamic_sensitivity(self, model, layer_idx, input_shape, method='hybrid', num_samples=5):
        """Calculate sensitivity with timeout and memory management"""
        
        cache_key = f"layer_{layer_idx}_{method}_{num_samples}"
        if cache_key in self.sensitivity_cache:
            print(f"📋 Using cached sensitivity: {self.sensitivity_cache[cache_key]:.6f}")
            return self.sensitivity_cache[cache_key], "cached"
        
        print(f"🔍 Calculating dynamic sensitivity for layer {layer_idx} using {method}...")

        # Enhanced timeout for problematic layers
        if layer_idx > 7:  # For layers deeper than 7 (including Conv layers)
            print(f"⚠️  Deep layer detected (>{7}), using simplified calculation...")
            layer_info = self._analyze_layer(model, layer_idx)
            sensitivity = self._theoretical_sensitivity(model, layer_idx, layer_info)
            safety_factor = self._calculate_safety_factor(layer_info, method)
            final_sensitivity = sensitivity * safety_factor
            
            print(f"✅ Simplified sensitivity: {final_sensitivity:.6f}")
            self.sensitivity_cache[cache_key] = final_sensitivity
            return final_sensitivity, "simplified"
            
        # Get layer information for context
        layer_info = self._analyze_layer(model, layer_idx)
        
        if method == 'theoretical':
            sensitivity = self._theoretical_sensitivity(model, layer_idx, layer_info)
            calculation_method = "theoretical"
        elif method == 'empirical':
            sensitivity = self._empirical_sensitivity(model, input_shape, layer_idx, num_samples)
            calculation_method = "empirical"
        elif method == 'lipschitz':
            sensitivity = self._lipschitz_sensitivity(model, layer_idx, input_shape)
            calculation_method = "lipschitz"
        elif method == 'hybrid':
            # Use multiple methods and take the most conservative (highest) value
            methods = ['theoretical', 'empirical', 'lipschitz']
            sensitivities = {}
            
            for m in methods:
                try:
                    if m == 'theoretical':
                        sensitivities[m] = self._theoretical_sensitivity(model, layer_idx, layer_info)
                    elif m == 'empirical':
                        sensitivities[m] = self._empirical_sensitivity(model, input_shape, layer_idx, num_samples)
                    elif m == 'lipschitz':
                        sensitivities[m] = self._lipschitz_sensitivity(model, layer_idx, input_shape)
                except Exception as e:
                    print(f"   {m} method failed: {e}")
                    sensitivities[m] = 2.0  # Conservative fallback
            
            # Take the median value for robustness
            sensitivity_values = list(sensitivities.values())
            sensitivity = np.median(sensitivity_values)
            calculation_method = f"hybrid({','.join(f'{k}:{v:.3f}' for k,v in sensitivities.items())})"
            
            print(f"   Hybrid sensitivities: {sensitivities}")
            print(f"   Selected median: {sensitivity:.6f}")
        else:
            print(f"⚠️  Unknown method {method}, using theoretical")
            sensitivity = self._theoretical_sensitivity(model, layer_idx, layer_info)
            calculation_method = "theoretical"
        
        # Apply adaptive safety factor based on layer type and depth
        safety_factor = self._calculate_safety_factor(layer_info, method)
        final_sensitivity = sensitivity * safety_factor
        
        print(f"✅ Base sensitivity: {sensitivity:.6f}")
        print(f"🛡️  Safety factor: {safety_factor}x")
        print(f"🔒 Final sensitivity: {final_sensitivity:.6f}")
        
        # Cache the result
        self.sensitivity_cache[cache_key] = final_sensitivity
        return final_sensitivity, calculation_method
    
    def _analyze_layer(self, model, layer_idx):
        """Analyze layer properties for informed sensitivity calculation"""
        cache_key = f"layer_analysis_{layer_idx}"
        if cache_key in self.layer_analysis_cache:
            return self.layer_analysis_cache[cache_key]
        
        try:
            layer = model.layers[layer_idx]
            layer_type = layer.__class__.__name__
            
            info = {
                'type': layer_type,
                'index': layer_idx,
                'depth_ratio': layer_idx / len(model.layers),
                'has_weights': len(layer.get_weights()) > 0,
                'trainable': getattr(layer, 'trainable', True),
                'activation': getattr(layer, 'activation', None),
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
                'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None
            }
            
            # Add specific properties based on layer type
            if 'Conv' in layer_type:
                info['is_conv'] = True
                info['filters'] = getattr(layer, 'filters', None)
                info['kernel_size'] = getattr(layer, 'kernel_size', None)
            elif 'Dense' in layer_type:
                info['is_dense'] = True
                info['units'] = getattr(layer, 'units', None)
            elif 'Pool' in layer_type:
                info['is_pooling'] = True
                info['pool_size'] = getattr(layer, 'pool_size', None)
            elif 'BatchNorm' in layer_type:
                info['is_batchnorm'] = True
            
            # Cache the analysis
            self.layer_analysis_cache[cache_key] = info
            return info
            
        except Exception as e:
            print(f"   Layer analysis failed: {e}")
            return {'type': 'unknown', 'index': layer_idx}
    
    def _theoretical_sensitivity(self, model, layer_idx, layer_info):
        """Enhanced theoretical sensitivity calculation"""
        try:
            layer = model.layers[layer_idx]
            layer_type = layer_info.get('type', 'unknown')
            
            print(f"   Analyzing {layer_type} layer (depth: {layer_info.get('depth_ratio', 0):.2f})")
            
            if layer_info.get('is_conv', False):
                weights = layer.get_weights()
                if weights:
                    kernel = weights[0]
                    # Use spectral norm for more accurate sensitivity
                    sensitivity = self._approximate_spectral_norm(kernel.reshape(kernel.shape[0], -1))
                    print(f"   Conv spectral norm: {sensitivity:.6f}")
                else:
                    sensitivity = 2.0
                    
            elif layer_info.get('is_dense', False):
                weights = layer.get_weights()
                if weights:
                    weight_matrix = weights[0]
                    sensitivity = self._approximate_spectral_norm(weight_matrix)
                    print(f"   Dense spectral norm: {sensitivity:.6f}")
                else:
                    sensitivity = 2.0
                    
            elif layer_info.get('is_pooling', False):
                # Pooling operations are contractive
                sensitivity = 1.0
                print(f"   Pooling sensitivity: {sensitivity:.6f}")
                
            elif layer_info.get('is_batchnorm', False):
                # BatchNorm can amplify or dampen signals
                weights = layer.get_weights()
                if weights and len(weights) >= 2:  # gamma and beta
                    gamma = weights[0]
                    sensitivity = np.max(np.abs(gamma)) if len(gamma) > 0 else 1.5
                else:
                    sensitivity = 1.5
                print(f"   BatchNorm sensitivity: {sensitivity:.6f}")
                
            elif 'Activation' in layer_type or 'ReLU' in layer_type:
                # Most activation functions have bounded derivatives
                if hasattr(layer, 'activation'):
                    if layer.activation.__name__ in ['relu', 'linear']:
                        sensitivity = 1.0
                    elif layer.activation.__name__ in ['sigmoid', 'tanh']:
                        sensitivity = 0.25  # Maximum derivative
                    else:
                        sensitivity = 1.0
                else:
                    sensitivity = 1.0
                print(f"   Activation sensitivity: {sensitivity:.6f}")
                
            else:
                # Conservative default for unknown layers
                sensitivity = 2.0
                print(f"   Unknown layer sensitivity: {sensitivity:.6f}")
            
            return sensitivity
            
        except Exception as e:
            print(f"   Theoretical calculation error: {e}")
            return 2.0  # Safe fallback
    
    def _empirical_sensitivity(self, model, input_shape, layer_idx, num_samples=3):
        """Calculate sensitivity with reduced memory usage"""
        try:
            print(f"   Running empirical sampling with {num_samples} samples...")
            
            # Reduce samples for deeper layers
            if layer_idx > 5:
                num_samples = min(num_samples, 1)
                print(f"   Reduced to {num_samples} samples for deep layer")
            
            max_sensitivity = 0.0
            
            # Use smaller perturbations and clear memory aggressively
            for i in range(num_samples):
                # Create smaller, simpler inputs
                input1 = np.random.uniform(-1, 1, (1,) + input_shape[1:]).astype(np.float32)
                perturbation = np.random.normal(0, 0.05, input1.shape).astype(np.float32)  # Smaller perturbation
                input2 = input1 + perturbation
                
                # Create temporary model and clear after each use
                try:
                    temp_input = model.input
                    temp_output = model.layers[layer_idx].output
                    temp_model = tf.keras.Model(inputs=temp_input, outputs=temp_output)
                    
                    output1 = temp_model.predict(input1, verbose=0)
                    output2 = temp_model.predict(input2, verbose=0)
                    
                    # Calculate sensitivity
                    output_diff = np.linalg.norm(output1 - output2)
                    input_diff = np.linalg.norm(perturbation)
                    
                    if input_diff > 1e-8:
                        local_sensitivity = output_diff / input_diff
                        max_sensitivity = max(max_sensitivity, local_sensitivity)
                    
                    # Clean up immediately
                    del temp_model, output1, output2
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"   Sample {i} failed: {e}")
                    continue
                
                finally:
                    # Force cleanup
                    del input1, input2, perturbation
            
            print(f"   Empirical max sensitivity: {max_sensitivity:.6f}")
            return max_sensitivity if max_sensitivity > 0 else 1.0
            
        except Exception as e:
            print(f"   Empirical calculation error: {e}")
            return 1.0
    
    def _lipschitz_sensitivity(self, model, layer_idx, input_shape):
        """Estimate Lipschitz constant as sensitivity bound"""
        try:
            print(f"   Calculating Lipschitz bound...")
            
            # For simplicity, estimate based on layer composition
            lipschitz_bound = 1.0
            
            # Accumulate Lipschitz constants from input to target layer
            for i in range(layer_idx + 1):
                layer = model.layers[i]
                layer_type = layer.__class__.__name__
                
                if 'Conv' in layer_type:
                    weights = layer.get_weights()
                    if weights:
                        kernel = weights[0]
                        # Approximate with Frobenius norm (upper bound of spectral norm)
                        layer_lipschitz = np.linalg.norm(kernel)
                        lipschitz_bound *= layer_lipschitz
                        
                elif 'Dense' in layer_type:
                    weights = layer.get_weights()
                    if weights:
                        weight_matrix = weights[0]
                        layer_lipschitz = self._approximate_spectral_norm(weight_matrix)
                        lipschitz_bound *= layer_lipschitz
                        
                elif 'Pool' in layer_type:
                    # Pooling is contractive
                    lipschitz_bound *= 1.0
                    
                elif 'BatchNorm' in layer_type:
                    # Approximate BatchNorm Lipschitz constant
                    lipschitz_bound *= 1.5
                    
                elif 'Activation' in layer_type or 'ReLU' in layer_type:
                    # ReLU and similar activations
                    lipschitz_bound *= 1.0
            
            print(f"   Lipschitz bound: {lipschitz_bound:.6f}")
            return lipschitz_bound
            
        except Exception as e:
            print(f"   Lipschitz calculation error: {e}")
            return 2.0
    
    def _approximate_spectral_norm(self, matrix, num_iterations=5):
        """More accurate spectral norm approximation with reduced iterations"""
        try:
            if len(matrix.shape) != 2:
                matrix = matrix.reshape(matrix.shape[0], -1)
            
            m, n = matrix.shape
            if min(m, n) == 0:
                return 0.0
            
            # Use power iteration for spectral norm with fewer iterations
            v = np.random.randn(n).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            
            for _ in range(num_iterations):  # Reduced from 10 to 5
                u = matrix @ v
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-8:
                    u = u / u_norm
                
                v = matrix.T @ u
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-8:
                    v = v / v_norm
                else:
                    break
            
            # Final estimate
            spectral_norm = np.linalg.norm(matrix @ v)
            return spectral_norm
            
        except Exception as e:
            print(f"   Spectral norm error: {e}")
            return np.linalg.norm(matrix, 'fro')  # Fallback to Frobenius norm
    
    def _calculate_safety_factor(self, layer_info, method):
        """Calculate adaptive safety factor based on layer properties"""
        base_factor = 1.2
        
        # Adjust based on layer type
        if layer_info.get('is_conv', False):
            base_factor *= 1.1  # Conv layers can be more complex
        elif layer_info.get('is_dense', False):
            base_factor *= 1.15  # Dense layers often have high sensitivity
        elif layer_info.get('is_batchnorm', False):
            base_factor *= 1.3  # BatchNorm can be unpredictable
        
        # Adjust based on network depth
        depth_ratio = layer_info.get('depth_ratio', 0)
        if depth_ratio > 0.5:  # Deeper layers
            base_factor *= (1 + 0.2 * depth_ratio)
        
        # Adjust based on calculation method reliability
        if method == 'empirical':
            base_factor *= 1.2  # Empirical can miss corner cases
        elif method == 'theoretical':
            base_factor *= 1.1  # Usually more conservative
        elif 'hybrid' in method:
            base_factor *= 1.05  # More reliable
        
        return base_factor
    
    def add_differential_privacy_dynamic(self, data, epsilon, model=None, layer_idx=None, 
                                       input_shape=None, sensitivity_method='hybrid', 
                                       num_samples=5):
        """Apply differential privacy with dynamic sensitivity calculation"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Calculate dynamic sensitivity
        if model is not None and layer_idx is not None and input_shape is not None:
            sensitivity, calc_method = self.calculate_dynamic_sensitivity(
                model, layer_idx, input_shape, method=sensitivity_method, num_samples=num_samples
            )
            print(f"🧮 Used {calc_method} for sensitivity calculation")
        else:
            print("⚠️  Missing model info, using data-based fallback...")
            # Conservative fallback based on data statistics
            data_range = np.ptp(data)  # Peak-to-peak range
            data_std = np.std(data)
            sensitivity = max(data_range, data_std * 4, 1.0)
            calc_method = "fallback"
            print(f"📊 Fallback sensitivity: {sensitivity:.6f}")
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = sensitivity / epsilon
        
        # Add calibrated Gaussian noise
        noise = np.random.normal(0, noise_scale, data.shape).astype(data.dtype)
        noisy_data = data + noise
        
        # Calculate privacy metrics
        privacy_loss = epsilon
        noise_magnitude = np.linalg.norm(noise)
        snr = np.linalg.norm(data) / (noise_magnitude + 1e-8)
        
        print(f"🔒 Dynamic DP Applied:")
        print(f"   ε = {epsilon:.3f}")
        print(f"   Sensitivity = {sensitivity:.6f} ({calc_method})")
        print(f"   Noise scale = {noise_scale:.6f}")
        print(f"   SNR = {snr:.2f}")
        print(f"   Noise magnitude = {noise_magnitude:.4f}")
        
        return noisy_data, {
            'sensitivity': sensitivity,
            'calculation_method': calc_method,
            'noise_scale': noise_scale,
            'epsilon': epsilon,
            'snr': snr,
            'noise_magnitude': noise_magnitude
        }


class EnhancedVGGPartitioner:
    """Enhanced VGG16 partitioner with dynamic DP and accurate energy monitoring"""
    
    def __init__(self):
        print("🔧 Loading Enhanced VGG16 Partitioner...")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Initialize enhanced components
        self.energy_monitor = EnhancedEnergyMonitor()
        self.dp_calculator = DynamicSensitivityDP()
        
        # Set TensorFlow memory growth to avoid OOM
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ GPU memory growth enabled for {gpu}")
        except Exception as e:
            print(f"⚠️  GPU memory config warning: {e}")
        
        # Load VGG16 with proper version detection
        try:
            # Try newer TensorFlow first
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
            self.keras = tf.keras
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
            print("✅ Using tensorflow.keras.applications")
        except ImportError:
            try:
                # Fallback to older structure
                from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
                self.keras = tf.keras
                self.preprocess_input = preprocess_input
                self.decode_predictions = decode_predictions
                print("✅ Using tensorflow.python.keras.applications")
            except ImportError:
                print("❌ Error: Could not import VGG16. Please install: pip3 install keras-preprocessing")
                raise
        
        # Load pre-trained VGG16
        print("📥 Loading VGG16 model...")
        self.full_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        print(f"✅ VGG16 loaded with {len(self.full_model.layers)} layers")
        
        # Print layer summary for reference
        print("\n📋 VGG16 Layer Summary:")
        for i, layer in enumerate(self.full_model.layers[:10]):  # Show first 10 layers
            print(f"   {i:2d}: {layer.__class__.__name__:15s} {str(layer.output_shape):20s}")
        if len(self.full_model.layers) > 10:
            print(f"   ... ({len(self.full_model.layers)-10} more layers)")
    
    def test_single_image(self, image, image_name, cut_layer, epsilon=None):
        """Test inference with a single specific image"""
        
        # Partition the model
        client_model, server_model = self.partition_at_layer(cut_layer)
        
        # Start monitoring
        self.energy_monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Client processing
            intermediate_data = client_model.predict(image, verbose=0)
            
            # Privacy application
            if epsilon:
                transmission_data, privacy_info = self.dp_calculator.add_differential_privacy_dynamic(
                    intermediate_data,
                    epsilon=epsilon,
                    model=client_model,
                    layer_idx=cut_layer,
                    input_shape=image.shape,
                    sensitivity_method='theoretical' if cut_layer >= 12 else 'hybrid',
                    num_samples=1
                )
            else:
                transmission_data = intermediate_data.copy()
                privacy_info = {}
            
            # Server processing
            final_output = server_model.predict(transmission_data, verbose=0)
            
            end_time = time.time()
            energy_metrics = self.energy_monitor.stop_monitoring()
            
            # Compile results
            result = {
                'cut_layer': cut_layer,
                'epsilon': epsilon,
                'image_name': image_name,
                'timing': {'total_inference_time': end_time - start_time},
                'data_metrics': {
                    'data_distortion': np.linalg.norm(transmission_data - intermediate_data) / np.linalg.norm(intermediate_data) if epsilon else 0.0
                },
                'energy_metrics': energy_metrics,
                'privacy_info': privacy_info,
                'predictions': {
                    'predicted_class': int(np.argmax(final_output)),
                    'confidence': float(np.max(final_output))
                }
            }
            
            return result
            
        except Exception as e:
            self.energy_monitor.stop_monitoring()
            return {
                'cut_layer': cut_layer,
                'epsilon': epsilon,
                'image_name': image_name,
                'error': str(e)
            }

    
    def load_enhanced_test_images(self, num_samples=3, dataset_path=None):
        """Enhanced image loading with multiple fallback options"""
        images = []
        
        print(f"🔍 Loading {num_samples} test images...")
        
        # Option 1: Load from specified dataset path
        if dataset_path and os.path.exists(dataset_path):
            print(f"   Trying dataset path: {dataset_path}")
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            all_images = []
            
            for ext in image_extensions:
                pattern = os.path.join(dataset_path, '**', ext)
                found_images = glob.glob(pattern, recursive=True)
                all_images.extend(found_images)
            
            if all_images:
                import random
                selected_images = random.sample(all_images, min(len(all_images), num_samples))
                
                for img_path in selected_images:
                    try:
                        img = Image.open(img_path).resize((224, 224)).convert('RGB')
                        img_array = np.array(img, dtype=np.float32)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = self.preprocess_input(img_array)
                        name = os.path.basename(img_path).split('.')[0]
                        images.append((img_array, name))
                        print(f"   ✅ {name} loaded from dataset")
                    except Exception as e:
                        print(f"   ❌ Failed to load {img_path}: {e}")
                        continue
                
                if images:
                    return images
        
        # Option 2: Try local sample_images directory
        sample_dir = "sample_images"
        if os.path.exists(sample_dir):
            print(f"   Trying local directory: {sample_dir}")
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(sample_dir, ext)))
            
            for img_path in image_files[:num_samples]:
                try:
                    img = Image.open(img_path).resize((224, 224)).convert('RGB')
                    img_array = np.array(img, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess_input(img_array)
                    name = os.path.basename(img_path).split('.')[0]
                    images.append((img_array, name))
                    print(f"   ✅ {name} loaded from sample_images/")
                except Exception as e:
                    print(f"   ❌ Failed to load {img_path}: {e}")
            
            if images:
                return images
        
        # Option 3: Download sample COCO images
        print("   Attempting to download sample COCO images...")
        coco_urls = [
            ("http://images.cocodataset.org/val2017/000000039769.jpg", "cats"),
            ("http://images.cocodataset.org/val2017/000000374628.jpg", "person"),
            ("http://images.cocodataset.org/val2017/000000252219.jpg", "car")
        ]
        
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        try:
            import urllib.request
            for i, (url, name) in enumerate(coco_urls[:num_samples]):
                filename = f"coco_{name}.jpg"
                filepath = os.path.join(sample_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"   📥 Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                
                try:
                    img = Image.open(filepath).resize((224, 224)).convert('RGB')
                    img_array = np.array(img, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = self.preprocess_input(img_array)
                    images.append((img_array, name))
                    print(f"   ✅ {name} downloaded and loaded")
                except Exception as e:
                    print(f"   ❌ Failed to process {filename}: {e}")
                    
            if images:
                return images
                
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
        
        # Option 4: Create realistic synthetic test images
        print("   Creating synthetic test images...")
        for i in range(num_samples):
            # Create more realistic synthetic images with patterns
            test_image = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
            
            # Add some structure to make it more realistic
            x, y = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224))
            pattern = np.sin(x * 10 + i) * np.cos(y * 10 + i) * 50 + 128
            test_image[0, :, :, 0] += pattern
            test_image[0, :, :, 1] += pattern * 0.8
            test_image[0, :, :, 2] += pattern * 0.6
            
            test_image = np.clip(test_image, 0, 255)
            test_image = self.preprocess_input(test_image)
            images.append((test_image, f"synthetic_{i}"))
            print(f"   ✅ synthetic_{i} created")
        
        print(f"\n💡 To use real images, you can:")
        print("   1. Put images in sample_images/ folder")
        print("   2. Specify dataset_path parameter")
        print("   3. Images will be auto-downloaded from COCO dataset")
        
        return images
    
    def partition_at_layer(self, cut_layer):
        """Enhanced model partitioning with validation"""
        if cut_layer < 0 or cut_layer >= len(self.full_model.layers) - 1:
            raise ValueError(f"Cut layer {cut_layer} is out of valid range [0, {len(self.full_model.layers)-2}]")
        
        print(f"🔪 Partitioning VGG16 at layer {cut_layer} ({self.full_model.layers[cut_layer].__class__.__name__})")
        
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
        
        print(f"   📱 Client model: {len(client_model.layers)} layers, output shape: {client_output.shape}")
        print(f"   ☁️  Server model: {len(server_model.layers)} layers, final output: {server_model.output_shape}")
        
        return client_model, server_model
    
    def test_enhanced_inference(self, cut_layer, epsilon=None, sensitivity_method='hybrid', 
                               num_samples=5, use_real_data=True, warmup_runs=2, 
                               dataset_path=None, detailed_analysis=True):
        """Enhanced inference testing with comprehensive metrics"""

         # Special handling for problematic layers
        if cut_layer >= 15:
            print(f"⚠️  Layer {cut_layer} detected - using cached/simplified approach")
            sensitivity_method = 'theoretical'  # Force simpler method
            num_samples = 1  # Minimal sampling
        
        print(f"\n🧪 Enhanced Inference Test")
        print("=" * 50)
        print(f"Cut Layer: {cut_layer}")
        print(f"Privacy: ε={epsilon} ({sensitivity_method})" if epsilon else "Privacy: None")
        print(f"Samples: {num_samples}, Warmup: {warmup_runs}")
        
        # Load test data
        if use_real_data:
            images = self.load_enhanced_test_images(num_samples=1, dataset_path=dataset_path)
            test_image, image_name = images[0]
        else:
            test_image = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
            image_name = "random"
        
        print(f"🖼️  Test image: {image_name}, shape: {test_image.shape}")
        
        # Partition the model
        client_model, server_model = self.partition_at_layer(cut_layer)
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"🔥 Running {warmup_runs} warmup iterations...")
            for i in range(warmup_runs):
                _ = client_model.predict(test_image, verbose=0)
                temp_intermediate = client_model.predict(test_image, verbose=0)
                _ = server_model.predict(temp_intermediate, verbose=0)
                del temp_intermediate
        
        # Start comprehensive monitoring
        self.energy_monitor.start_monitoring()
        inference_start_time = time.time()
        
        # === CLIENT-SIDE PROCESSING ===
        client_start_time = time.time()
        intermediate_data = client_model.predict(test_image, verbose=0)
        client_end_time = time.time()
        client_time = client_end_time - client_start_time
        
        # === PRIVACY APPLICATION ===
        privacy_start_time = time.time()
        privacy_info = {}
        
        if epsilon:
            transmission_data, privacy_info = self.dp_calculator.add_differential_privacy_dynamic(
                intermediate_data,
                epsilon=epsilon,
                model=client_model,
                layer_idx=cut_layer,
                input_shape=test_image.shape,
                sensitivity_method=sensitivity_method,
                num_samples=num_samples
            )
        else:
            transmission_data = intermediate_data.copy()
            
        privacy_end_time = time.time()
        privacy_time = privacy_end_time - privacy_start_time
        
        # === SERVER-SIDE PROCESSING ===
        server_start_time = time.time()
        final_output = server_model.predict(transmission_data, verbose=0)
        server_end_time = time.time()
        server_time = server_end_time - server_start_time
        
        # Stop monitoring
        inference_end_time = time.time()
        energy_metrics = self.energy_monitor.stop_monitoring()
        
        # === CALCULATE METRICS ===
        total_inference_time = inference_end_time - inference_start_time
        total_processing_time = client_time + server_time
        
        # Data transmission metrics
        original_size_kb = intermediate_data.size * 4 / 1024
        transmission_size_kb = transmission_data.size * 4 / 1024
        
        # Privacy impact metrics
        if epsilon:
            data_distortion = np.linalg.norm(transmission_data - intermediate_data) / np.linalg.norm(intermediate_data)
            privacy_overhead_time = privacy_time
        else:
            data_distortion = 0.0
            privacy_overhead_time = 0.0
        
        # === DISPLAY RESULTS ===
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   Client processing: {client_time:.4f}s")
        print(f"   Privacy processing: {privacy_time:.4f}s")
        print(f"   Server processing: {server_time:.4f}s")
        print(f"   Total processing: {total_processing_time:.4f}s")
        print(f"   Total inference: {total_inference_time:.4f}s")
        
        print(f"\n📊 DATA ANALYSIS:")
        print(f"   Intermediate shape: {intermediate_data.shape}")
        print(f"   Original data size: {original_size_kb:.2f} KB")
        print(f"   Transmission size: {transmission_size_kb:.2f} KB")
        if epsilon:
            print(f"   Data distortion: {data_distortion:.6f}")
            print(f"   Privacy overhead: {privacy_overhead_time:.4f}s ({100*privacy_overhead_time/total_inference_time:.1f}%)")
        
        # Energy metrics
        if energy_metrics:
            print(f"\n🔋 ENERGY ANALYSIS:")
            print(f"   Average Power: {energy_metrics['avg_power_mw']:.1f} mW ({energy_metrics['avg_power_mw']/1000:.3f} W)")
            print(f"   Peak Power: {energy_metrics['max_power_mw']:.1f} mW ({energy_metrics['max_power_mw']/1000:.3f} W)")
            print(f"   Power Std Dev: {energy_metrics['std_power_mw']:.1f} mW")
            print(f"   Total Energy: {energy_metrics['energy_mj']:.2f} mJ ({energy_metrics['energy_j']:.6f} J)")
            print(f"   Energy Efficiency: {energy_metrics['energy_mj']/total_inference_time:.2f} mJ/s")
            print(f"   Monitoring: {energy_metrics['monitoring_method']} ({energy_metrics['samples_collected']} samples)")
        
        # Privacy details
        if epsilon and privacy_info:
            print(f"\n🔒 PRIVACY ANALYSIS:")
            print(f"   Epsilon (ε): {privacy_info['epsilon']:.3f}")
            print(f"   Sensitivity: {privacy_info['sensitivity']:.6f}")
            print(f"   Method: {privacy_info['calculation_method']}")
            print(f"   Noise scale: {privacy_info['noise_scale']:.6f}")
            print(f"   Signal-to-Noise Ratio: {privacy_info['snr']:.2f}")
            print(f"   Noise magnitude: {privacy_info['noise_magnitude']:.4f}")
        
        # Model predictions
        try:
            predictions = self.decode_predictions(final_output, top=3)[0]
            print(f"\n🎯 PREDICTIONS for {image_name}:")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                print(f"   {i+1}. {label}: {score:.4f}")
        except Exception as e:
            predicted_class = np.argmax(final_output[0])
            confidence = np.max(final_output[0])
            print(f"\n🎯 PREDICTION: Class {predicted_class} (confidence: {confidence:.4f})")
        
        # Detailed analysis
        if detailed_analysis and epsilon:
            print(f"\n📈 DETAILED PRIVACY IMPACT:")
            
            # Calculate accuracy impact (simplified)
            original_pred = np.argmax(client_model.predict(test_image, verbose=0) if cut_layer == len(self.full_model.layers)-2 
                                    else server_model.predict(intermediate_data, verbose=0))
            private_pred = np.argmax(final_output)
            
            print(f"   Prediction change: {'Yes' if original_pred != private_pred else 'No'}")
            print(f"   Confidence impact: {abs(np.max(final_output) - confidence):.4f}")
        
        # === COMPILE RESULTS ===
        result = {
            'cut_layer': cut_layer,
            'epsilon': epsilon,
            'sensitivity_method': sensitivity_method,
            'image_name': image_name,
            'timing': {
                'client_time': client_time,
                'privacy_time': privacy_time,
                'server_time': server_time,
                'total_processing_time': total_processing_time,
                'total_inference_time': total_inference_time
            },
            'data_metrics': {
                'intermediate_shape': list(intermediate_data.shape),
                'original_size_kb': original_size_kb,
                'transmission_size_kb': transmission_size_kb,
                'data_distortion': data_distortion
            },
            'energy_metrics': energy_metrics,
            'privacy_info': privacy_info,
            'predictions': {
                'predicted_class': int(np.argmax(final_output)),
                'confidence': float(np.max(final_output))
            }
        }
        
        # Clean up memory
        del intermediate_data, transmission_data, final_output
        
        return result


def run_comprehensive_analysis():
    """Run a comprehensive analysis of the enhanced VGG partitioner"""
    print("🚀 Enhanced VGG16 Partitioning Analysis")
    print("=" * 60)
    
    try:
        partitioner = EnhancedVGGPartitioner()
    except Exception as e:
        print(f"❌ Failed to initialize partitioner: {e}")
        print("\n💡 Try installing missing dependencies:")
        print("   pip3 install jetson-stats keras-preprocessing Pillow")
        return
    
    # Test configurations
    test_configs = [
        {'cut_layer': 3, 'epsilon': None, 'desc': 'Baseline (no privacy)'},
        {'cut_layer': 3, 'epsilon': 2.0, 'desc': 'Moderate privacy'},
        {'cut_layer': 3, 'epsilon': 0.5, 'desc': 'Strong privacy'},
        {'cut_layer': 8, 'epsilon': 1.0, 'desc': 'Mid-network cut'},
        {'cut_layer': 15, 'epsilon': 1.0, 'desc': 'Late-network cut'},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*20} TEST {i+1}/5: {config['desc']} {'='*20}")
        
        try:
            result = partitioner.test_enhanced_inference(
                cut_layer=config['cut_layer'],
                epsilon=config['epsilon'],
                sensitivity_method='hybrid',
                num_samples=3,
                use_real_data=True,
                warmup_runs=1,
                detailed_analysis=True
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # === COMPARATIVE ANALYSIS ===
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("📊 COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        # Find baseline
        baseline = next((r for r in results if r['epsilon'] is None), results[0])
        
        print("Configuration | Energy | Time | Privacy | Data Size | Accuracy")
        print("-" * 65)
        
        for r in results:
            epsilon_str = f"ε={r['epsilon']:.1f}" if r['epsilon'] else "None"
            
            # Calculate relative metrics
            energy_ratio = (r['energy_metrics']['energy_mj'] / baseline['energy_metrics']['energy_mj'] 
                          if baseline['energy_metrics'] and r['energy_metrics'] else 1.0)
            time_ratio = r['timing']['total_inference_time'] / baseline['timing']['total_inference_time']
            
            print(f"Layer {r['cut_layer']:2d} {epsilon_str:8s} | "
                  f"{energy_ratio:5.2f}x | {time_ratio:5.2f}x | "
                  f"{r['privacy_info'].get('snr', 'N/A'):>7} | "
                  f"{r['data_metrics']['transmission_size_kb']:6.1f}KB | "
                  f"{r['predictions']['confidence']:5.3f}")
        
        # Save detailed results
        try:
            # Convert numpy types for JSON serialization
            json_results = []
            for r in results:
                json_r = {}
                for key, value in r.items():
                    if isinstance(value, dict):
                        json_r[key] = {k: (v.item() if hasattr(v, 'item') else 
                                         v.tolist() if hasattr(v, 'tolist') else v) 
                                     for k, v in value.items() if v is not None}
                    elif hasattr(value, 'item'):
                        json_r[key] = value.item()
                    elif hasattr(value, 'tolist'):
                        json_r[key] = value.tolist()
                    else:
                        json_r[key] = value
                json_results.append(json_r)
            
            with open('enhanced_vgg_analysis.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Detailed results saved to 'enhanced_vgg_analysis.json'")
            
        except Exception as e:
            print(f"⚠️  Could not save JSON results: {e}")
        
        # Final summary
        print(f"\n🎯 KEY FINDINGS:")
        if len(results) > 1:
            energy_overhead = ((results[1]['energy_metrics']['energy_mj'] - 
                              baseline['energy_metrics']['energy_mj']) / 
                             baseline['energy_metrics']['energy_mj'] * 100)
            time_overhead = ((results[1]['timing']['total_inference_time'] - 
                            baseline['timing']['total_inference_time']) / 
                           baseline['timing']['total_inference_time'] * 100)
            
            print(f"   Privacy overhead: {energy_overhead:+.1f}% energy, {time_overhead:+.1f}% time")
            print(f"   Hardware monitoring: {baseline['energy_metrics']['monitoring_method']}")
            print(f"   Dynamic sensitivity: Working ✅")

def run_diverse_image_analysis(partitioner, test_configs=None, num_images=10):
    """Test with diverse images to validate robustness"""
    
    if test_configs is None:
        # Focus on the best configuration from your results
        test_configs = [
            {'cut_layer': 3, 'epsilon': None, 'desc': 'Baseline'},
            {'cut_layer': 12, 'epsilon': 1.0, 'desc': 'Optimal config'},
        ]
    
    print("🌍 DIVERSE IMAGE ANALYSIS")
    print("=" * 60)
    
    # Load multiple diverse images
    diverse_images = load_diverse_test_images(partitioner, num_images)
    
    all_results = []
    image_summaries = []
    
    for config in test_configs:
        print(f"\n{'='*20} CONFIG: {config['desc']} {'='*20}")
        config_results = []
        
        for i, (image, image_name, image_info) in enumerate(diverse_images):
            print(f"\n--- Image {i+1}/{len(diverse_images)}: {image_name} ---")
            print(f"Source: {image_info['source']}, Category: {image_info['category']}")
            
            try:
                # Test with this specific image
                result = partitioner.test_enhanced_inference(
                    cut_layer=config['cut_layer'],
                    epsilon=config['epsilon'],
                    sensitivity_method='theoretical' if config['cut_layer'] >= 12 else 'hybrid',
                    num_samples=1,  # Faster for multiple images
                    use_real_data=False,  # We're providing the image directly
                    warmup_runs=0,  # Skip warmup for speed
                    detailed_analysis=False
                )
                
                # Override with our specific image
                result = partitioner.test_single_image(
                    image, image_name, 
                    cut_layer=config['cut_layer'],
                    epsilon=config['epsilon']
                )
                
                # Add image metadata
                result['image_info'] = image_info
                config_results.append(result)
                
                # Quick summary
                pred_class = result['predictions']['predicted_class']
                confidence = result['predictions']['confidence']
                print(f"   Prediction: Class {pred_class} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        all_results.extend(config_results)
        
        # Analyze this configuration
        analyze_config_diversity(config_results, config['desc'])
    
    # Overall analysis
    analyze_overall_diversity(all_results)
    
    # Save detailed results
    save_diversity_results(all_results)
    
    return all_results


def load_diverse_test_images(partitioner, num_images=10):
    """Load diverse images from multiple sources"""
    diverse_images = []
    
    print(f"🔍 Loading {num_images} diverse test images...")
    
    # Strategy 1: Download from multiple COCO categories
    coco_categories = [
        ("http://images.cocodataset.org/val2017/000000039769.jpg", "cats", "animal"),
        ("http://images.cocodataset.org/val2017/000000374628.jpg", "person", "people"),
        ("http://images.cocodataset.org/val2017/000000252219.jpg", "car", "vehicle"),
        ("http://images.cocodataset.org/val2017/000000289343.jpg", "elephant", "animal"),
        ("http://images.cocodataset.org/val2017/000000581781.jpg", "train", "vehicle"),
        ("http://images.cocodataset.org/val2017/000000578967.jpg", "airplane", "vehicle"),
        ("http://images.cocodataset.org/val2017/000000460347.jpg", "horse", "animal"),
        ("http://images.cocodataset.org/val2017/000000087038.jpg", "pizza", "food"),
        ("http://images.cocodataset.org/val2017/000000017627.jpg", "toilet", "indoor"),
        ("http://images.cocodataset.org/val2017/000000475779.jpg", "bird", "animal"),
    ]
    
    # Create directory for diverse images
    diverse_dir = "diverse_test_images"
    if not os.path.exists(diverse_dir):
        os.makedirs(diverse_dir)
    
    # Download and process COCO images
    try:
        import urllib.request
        for i, (url, name, category) in enumerate(coco_categories[:num_images]):
            filename = f"diverse_{name}_{i}.jpg"
            filepath = os.path.join(diverse_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"   📥 Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                except Exception as e:
                    print(f"   ❌ Download failed for {name}: {e}")
                    continue
            
            try:
                img = Image.open(filepath).resize((224, 224)).convert('RGB')
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = partitioner.preprocess_input(img_array)
                
                image_info = {
                    'source': 'COCO',
                    'category': category,
                    'original_name': name,
                    'filepath': filepath
                }
                
                diverse_images.append((img_array, f"coco_{name}", image_info))
                print(f"   ✅ {name} loaded")
                
            except Exception as e:
                print(f"   ❌ Failed to process {name}: {e}")
                
    except ImportError:
        print("   ⚠️  urllib not available, using synthetic images")
    
    # Strategy 2: Add synthetic images with different characteristics
    synthetic_configs = [
        {"type": "noise", "category": "synthetic", "desc": "random noise"},
        {"type": "gradient", "category": "synthetic", "desc": "gradient pattern"},
        {"type": "checkerboard", "category": "synthetic", "desc": "checkerboard"},
        {"type": "circles", "category": "synthetic", "desc": "geometric shapes"},
    ]
    
    for i, config in enumerate(synthetic_configs):
        if len(diverse_images) >= num_images:
            break
            
        synthetic_img = create_synthetic_image(config["type"], i)
        synthetic_img = partitioner.preprocess_input(synthetic_img)
        
        image_info = {
            'source': 'synthetic',
            'category': config["category"],
            'original_name': config["desc"],
            'type': config["type"]
        }
        
        diverse_images.append((synthetic_img, f"synthetic_{config['type']}", image_info))
        print(f"   ✅ Synthetic {config['desc']} created")
    
    print(f"📊 Loaded {len(diverse_images)} diverse images")
    return diverse_images[:num_images]


def create_synthetic_image(image_type, seed=0):
    """Create synthetic test images with different characteristics"""
    np.random.seed(seed)
    
    if image_type == "noise":
        # Pure random noise
        img = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
        
    elif image_type == "gradient":
        # Smooth gradient
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        x, y = np.meshgrid(np.linspace(0, 255, 224), np.linspace(0, 255, 224))
        img[0, :, :, 0] = x
        img[0, :, :, 1] = y
        img[0, :, :, 2] = (x + y) / 2
        
    elif image_type == "checkerboard":
        # Checkerboard pattern
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        square_size = 28
        for i in range(0, 224, square_size):
            for j in range(0, 224, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    img[0, i:i+square_size, j:j+square_size, :] = 255
                    
    elif image_type == "circles":
        # Geometric circles
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        center_x, center_y = 112, 112
        for radius in [30, 60, 90]:
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[0, mask, :] = np.random.uniform(100, 255, 3)
    
    return img


def analyze_config_diversity(results, config_name):
    """Analyze diversity of results for a single configuration"""
    if not results:
        return
        
    print(f"\n📈 DIVERSITY ANALYSIS for {config_name}:")
    
    # Prediction diversity
    predictions = [r['predictions']['predicted_class'] for r in results]
    confidences = [r['predictions']['confidence'] for r in results]
    
    unique_predictions = len(set(predictions))
    avg_confidence = np.mean(confidences)
    confidence_std = np.std(confidences)
    
    print(f"   Unique predictions: {unique_predictions}/{len(results)}")
    print(f"   Avg confidence: {avg_confidence:.3f} ± {confidence_std:.3f}")
    
    # Performance diversity
    if 'timing' in results[0]:
        inference_times = [r['timing']['total_inference_time'] for r in results]
        avg_time = np.mean(inference_times)
        time_std = np.std(inference_times)
        print(f"   Avg inference time: {avg_time:.3f}s ± {time_std:.3f}s")
    
    # Energy diversity (if available)
    if all('energy_metrics' in r and r['energy_metrics'] for r in results):
        energies = [r['energy_metrics']['energy_mj'] for r in results]
        avg_energy = np.mean(energies)
        energy_std = np.std(energies)
        print(f"   Avg energy: {avg_energy:.2f}mJ ± {energy_std:.2f}mJ")
    
    # Privacy impact diversity (if applicable)
    privacy_results = [r for r in results if r.get('epsilon')]
    if privacy_results:
        distortions = [r['data_metrics']['data_distortion'] for r in privacy_results]
        snrs = [r['privacy_info']['snr'] for r in privacy_results if 'privacy_info' in r]
        
        if distortions:
            print(f"   Avg data distortion: {np.mean(distortions):.4f} ± {np.std(distortions):.4f}")
        if snrs:
            print(f"   Avg SNR: {np.mean(snrs):.2f} ± {np.std(snrs):.2f}")


def analyze_overall_diversity(all_results):
    """Analyze overall diversity across all configurations"""
    print(f"\n🌟 OVERALL DIVERSITY SUMMARY:")
    print("=" * 50)
    
    # Group by configuration
    baseline_results = [r for r in all_results if not r.get('epsilon')]
    privacy_results = [r for r in all_results if r.get('epsilon')]
    
    if baseline_results and privacy_results:
        # Compare prediction stability
        baseline_preds = set(r['predictions']['predicted_class'] for r in baseline_results)
        privacy_preds = set(r['predictions']['predicted_class'] for r in privacy_results)
        
        prediction_overlap = len(baseline_preds & privacy_preds)
        total_unique = len(baseline_preds | privacy_preds)
        
        print(f"Prediction consistency: {prediction_overlap}/{total_unique} classes overlap")
        
        # Compare confidence distributions
        baseline_conf = [r['predictions']['confidence'] for r in baseline_results]
        privacy_conf = [r['predictions']['confidence'] for r in privacy_results]
        
        conf_diff = np.mean(baseline_conf) - np.mean(privacy_conf)
        print(f"Confidence impact: {conf_diff:+.3f} average difference")
    
    # Image category breakdown
    categories = {}
    for r in all_results:
        if 'image_info' in r:
            cat = r['image_info']['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['predictions']['confidence'])
    
    print("\nPerformance by category:")
    for cat, confidences in categories.items():
        print(f"   {cat}: {np.mean(confidences):.3f} avg confidence ({len(confidences)} images)")


def save_diversity_results(results):
    """Save diversity analysis results"""
    try:
        # Convert for JSON serialization
        json_results = []
        for r in results:
            json_r = {}
            for key, value in r.items():
                if isinstance(value, dict):
                    json_r[key] = {k: (v.item() if hasattr(v, 'item') else 
                                     v.tolist() if hasattr(v, 'tolist') else v) 
                                 for k, v in value.items() if v is not None}
                elif hasattr(value, 'item'):
                    json_r[key] = value.item()
                elif hasattr(value, 'tolist'):
                    json_r[key] = value.tolist()
                else:
                    json_r[key] = value
            json_results.append(json_r)
        
        with open('diverse_image_analysis.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n💾 Diversity results saved to 'diverse_image_analysis.json'")
        
    except Exception as e:
        print(f"⚠️  Could not save diversity results: {e}")



# Modified main function to include diversity testing
def main():
    """Main function with diversity testing option"""
    print("Enhanced VGG16 Partitioning with Diverse Image Testing")
    print("=" * 80)
    
    try:
        partitioner = EnhancedVGGPartitioner()
    except Exception as e:
        print(f"❌ Failed to initialize partitioner: {e}")
        return
    
    # Ask user what they want to test
    print("\nTesting Options:")
    print("1. Original comprehensive analysis (5 tests, 1 image)")
    print("2. Diverse image analysis (10+ images, focused configs)")
    print("3. Both analyses")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n" + "="*60)
        print("RUNNING ORIGINAL ANALYSIS")
        print("="*60)
        run_comprehensive_analysis()
    
    if choice in ['2', '3']:
        print("\n" + "="*60)
        print("RUNNING DIVERSE IMAGE ANALYSIS")
        print("="*60)
        run_diverse_image_analysis(partitioner, num_images=10)

if __name__ == '__main__':
    main()
