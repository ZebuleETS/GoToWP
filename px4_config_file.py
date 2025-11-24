#!/usr/bin/env python3
"""
PX4 Parameter Update Script
Updates drone parameters based on UAV specifications
"""

def update_params(input_file, output_file):
    """
    Update PX4 parameters to match UAV specifications
    
    UAV Specifications:
    - Battery: 10.0 Ah (with 20% reserve = 2.0 Ah)
    - Empty weight: 1.6 kg
    - Max power: 775 W
    - Wing area: 0.5 m²
    - Aspect ratio: 15.7
    - Oswald efficiency: 0.85
    - Zero-lift drag: 0.0107
    - Max airspeed: 30.0 m/s
    - Min airspeed: 8.0 m/s
    - Max turn rate: 0.7 rad/s (≈40°/s)
    """
    
    # Parameter updates mapping
    param_updates = {
        # Battery parameters (10 Ah capacity, 4S LiPo)
        'BAT1_CAPACITY': 10.0,  # Ah
        'BAT1_N_CELLS': 4,      # 4S battery
        'BAT_LOW_THR': 0.20,    # 20% low threshold (reserve)
        'BAT_CRIT_THR': 0.10,   # 10% critical threshold
        'BAT_EMERGEN_THR': 0.05, # 5% emergency threshold
        
        # Airspeed parameters
        'FW_AIRSPD_MAX': 30.0,   # Max airspeed m/s
        'FW_AIRSPD_MIN': 8.0,    # Min airspeed m/s (stall + margin)
        'FW_AIRSPD_TRIM': 15.0,  # Cruise airspeed m/s
        'FW_AIRSPD_STALL': 7.0,  # Stall speed m/s
        
        # Aerodynamic parameters
        'FW_WING_SPAN': 2.805,   # Wing span m (calculated from area & AR)
        'FW_WING_HEIGHT': 0.5,   # Wing height m
        
        # Turn rate / Roll limits (max turn rate 0.7 rad/s ≈ 40°/s)
        'FW_R_LIM': 40.0,        # Max roll angle (degrees)
        'FW_R_RMAX': 40.0,       # Max roll rate (deg/s)
        'FW_P_RMAX_POS': 40.0,   # Max pitch rate positive (deg/s)
        'FW_P_RMAX_NEG': 40.0,   # Max pitch rate negative (deg/s)
        'FW_Y_RMAX': 30.0,       # Max yaw rate (deg/s)
        
        # Pitch limits
        'FW_P_LIM_MAX': 30.0,    # Max pitch angle (degrees)
        'FW_P_LIM_MIN': -15.0,   # Min pitch angle (degrees)
        
        # Throttle limits (based on max power 775W)
        'FW_THR_MAX': 0.95,      # Max throttle
        'FW_THR_MIN': 0.05,      # Min throttle
        'FW_THR_IDLE': 0.05,     # Idle throttle
        'FW_THR_TRIM': 0.60,     # Cruise throttle (adjusted for efficiency)
        
        # TECS (Total Energy Control System) parameters
        'FW_T_CLMB_MAX': 5.0,    # Max climb rate m/s
        'FW_T_SINK_MAX': 2.5,    # Max sink rate m/s
        'FW_T_SINK_MIN': 2.0,    # Min sink rate m/s
        
        # EKF2 Airspeed parameters
        'EKF2_ASPD_MAX': 30.0,   # Max airspeed for EKF
        'EKF2_ARSP_THR': 8.0,    # Airspeed threshold
        
        # Loiter radius (based on turn rate and speed)
        'NAV_LOITER_RAD': 60.0,  # Loiter radius m (adjusted for turn rate)
        
        # Landing parameters
        'FW_LND_AIRSPD': 10.0,   # Landing airspeed m/s
        'FW_LND_ANG': 5.0,       # Landing approach angle (degrees)
        
        # Takeoff parameters
        'FW_TKO_AIRSPD': 12.0,   # Takeoff airspeed m/s
        'FW_TKO_PITCH_MIN': 10.0, # Min takeoff pitch (degrees)
    }
    
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Update parameters
    updated_lines = []
    updates_made = {}
    
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('#') or line.strip() == '':
            updated_lines.append(line)
            continue
        
        # Parse parameter line
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            vehicle_id = parts[0]
            component_id = parts[1]
            param_name = parts[2]
            param_value = parts[3]
            param_type = parts[4] if len(parts) > 4 else '9'
            
            # Check if this parameter needs updating
            if param_name in param_updates:
                new_value = param_updates[param_name]
                updated_line = f"{vehicle_id}\t{component_id}\t{param_name}\t{new_value:.18f}\t{param_type}\n"
                updated_lines.append(updated_line)
                updates_made[param_name] = (param_value, new_value)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    # Write output file
    with open(output_file, 'w') as f:
        f.writelines(updated_lines)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PX4 PARAMETER UPDATE SUMMARY")
    print(f"{'='*60}\n")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"\nUpdated {len(updates_made)} parameters:\n")
    
    for param, (old_val, new_val) in sorted(updates_made.items()):
        print(f"  {param:20s}: {old_val:>15s} -> {new_val:>15.6f}")
    
    print(f"\n{'='*60}")
    print("\nDerived calculations:")
    print(f"  Wing span: {2.805:.3f} m (from area={0.5} m² and AR={15.7})")
    print(f"  Max turn rate: ~40°/s (0.7 rad/s)")
    print(f"  Battery reserve: 2.0 Ah (20% of 10.0 Ah)")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Update parameters
    update_params(
        'current_drone_params.params',
        'updated_drone_params.params'
    )
    
    print("✓ Parameter file updated successfully!")
    print("\nNext steps:")
    print("  1. Review updated_drone_params.params")
    print("  2. Upload to drone using QGroundControl or:")
    print("     px4-param load updated_drone_params.params")
    print("  3. Test in simulation before real flight")