import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

class SimpleBeam:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height
        self.loads = []
        self.I = (width * height**3) / 12  # Moment of inertia
        self.cracks = []  # Stores crack information    

    def add_point_load(self, load, position):
        self.loads.append({'type': 'point', 'load': load, 'position': position})

    def add_uniform_load(self, load, start, end):
        self.loads.append({'type': 'uniform', 'load': load, 'start': start, 'end': end})

    def add_moment_load(self, moment, position):
        self.loads.append({'type': 'moment', 'moment': moment, 'position': position})
        
    def delete_load(self, index):
        """Deletes a specified load."""
        if 0 <= index < len(self.loads):
            del self.loads[index]
        else:
            print("Invalid value.")

    def add_crack(self, crack_position, crack_height):
        self.cracks.append({'position': crack_position, 'height': crack_height}) 

    def calculate_reactions(self):
        total_load = 0
        moment_sum = 0

        for load in self.loads:
            if load['type'] == 'point':
                total_load += load['load']
                moment_sum += load['load'] * load['position']
            elif load['type'] == 'uniform':
                uniform_load = load['load'] * (load['end'] - load['start'])
                total_load += uniform_load
                moment_sum += uniform_load * (load['start'] + load['end']) / 2
            elif load['type'] == 'moment':
                moment_sum += load['moment']

        reaction_b = moment_sum / self.length
        reaction_a = total_load - reaction_b

        return round(reaction_a, 1), round(reaction_b, 1)

    def shear_force_and_bending_moment(self):
        reaction_a, reaction_b = self.calculate_reactions()
        x = np.linspace(0, self.length, 1000)
        shear_force = np.zeros_like(x)

        for i, xi in enumerate(x):
            shear = reaction_a
            for load in self.loads:
                if load['type'] == 'point' and xi >= load['position']:
                    shear -= load['load']
                elif load['type'] == 'uniform' and xi >= load['start']:
                    load_length = min(xi, load['end']) - load['start']
                    shear -= load['load'] * load_length

            if xi >= self.length:
                shear += reaction_b

            shear_force[i] = round(shear, 1)

        bending_moment = cumtrapz(shear_force, x, initial=0)

        for load in self.loads:
            if load['type'] == 'moment':
                idx = np.searchsorted(x, load['position'])
                bending_moment[idx:] += load['moment']

        bending_moment[0] = 0
        bending_moment[-1] = 0

        return x, shear_force, bending_moment
    
    def get_modified_properties(self, x_value, for_shear=False):                                             
        applicable_cracks = [crack for crack in self.cracks if crack['position'] <= x_value]
        if applicable_cracks and for_shear:
            crack = applicable_cracks[-1]
            crack_height = crack['height']
            new_height = self.height - crack_height
            new_I = (self.width * new_height**3) / 12
            new_y = new_height / 2
            return new_I, new_y
        return self.I, self.height / 2

    def get_shear_force_and_moment_at(self, x_value):
        x, shear_force, bending_moment = self.shear_force_and_bending_moment()
        idx = np.searchsorted(x, x_value)
        
        # Calculate left-hand and right-hand values
        shear_left = shear_force[idx - 1] if idx > 0 else shear_force[idx]
        shear_right = shear_force[idx] if idx < len(shear_force) - 1 else shear_force[idx - 1]
        moment_left = bending_moment[idx - 1] if idx > 0 else bending_moment[idx]
        moment_right = bending_moment[idx] if idx < len(bending_moment) - 1 else bending_moment[idx - 1]
        
        # Choose the greater value
        shear = max(shear_left, shear_right)
        moment = max(moment_left, moment_right)

        return round(shear, 1), round(moment, 1)

    def calculate_stress(self, x_value, y_value, z_value, shear=None, moment=None):
        if shear is None or moment is None:
            shear, moment = self.get_shear_force_and_moment_at(x_value)
        modified_I, modified_y = self.get_modified_properties(x_value, for_shear=True)

        # If y is less than crack height, set stress to 0
        applicable_cracks = [crack for crack in self.cracks if crack['position'] == x_value]
        if applicable_cracks:
            crack = applicable_cracks[-1]
            if y_value < (self.height * -0.5 + crack['height']):
                print(f'y = {y_value}: Stress is 0 due to crack at this position.')
                return 0.0, 0.0

        # Calculate modified y
        if applicable_cracks:
            crack = applicable_cracks[-1]
            y_value = y_value - (crack['height'] / 2)
            sigma = -(moment * y_value) / modified_I
        else:
            # Calculate sigma using default approach when no cracks are present
            sigma = -(moment * y_value) / self.I

        # Calculate bending stress and shear stress
        if applicable_cracks:
            crack = applicable_cracks[-1]
            tau = -(1.5 * shear) / (self.width * (self.height - crack['height'])) * (1 - (2 * y_value / (self.height - crack['height']))**2)
        else:
            tau = -(1.5 * shear) / (self.width * self.height) * (1 - (2 * y_value / self.height)**2)
            
        return round(sigma, 1), round(tau, 1)

    def calculate_stress_tensor(self, x_value, y_value, z_value):
        sigma, tau = self.calculate_stress(x_value, y_value, z_value)
        # Construct a 3x3 stress tensor
        stress_tensor = np.array([
            [sigma, tau, 0],
            [tau, 0, 0],
            [0, 0, 0]
        ])
        return stress_tensor

    def plot_results(self):
        x, shear_force, bending_moment = self.shear_force_and_bending_moment()

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(x, shear_force, label='Shear Force', color='b')
        plt.xlabel('Position along the beam (m)')
        plt.ylabel('Shear Force (kN)')
        plt.title('Shear Force Diagram')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, bending_moment, label='Bending Moment', color='r')
        plt.xlabel('Position along the beam (m)')
        plt.ylabel('Bending Moment (kNm)')
        plt.title('Bending Moment Diagram')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_mohrs_circle(sigma_ut, sigma_uc, tau_us, specific_sigma, specific_tau):
    """
    Draws four types of Mohr's Circles.
    :param sigma_ut: Ultimate tensile stress
    :param sigma_uc: Ultimate compressive stress
    :param tau_us: Ultimate shear stress
    :param specific_sigma: Normal stress calculated at a specific point
    :param specific_tau: Shear stress calculated at a specific point
    """
    # Define circles
    theta = np.linspace(0, 2 * np.pi, 500)  # Angle for parametric circle
    
    # Mohr's Circle for Tensile
    center_tensile = sigma_ut / 2
    radius_tensile = sigma_ut / 2
    x_tensile = center_tensile + radius_tensile * np.cos(theta)
    y_tensile = radius_tensile * np.sin(theta)
    
    # Mohr's Circle for Compressive
    center_compressive = - sigma_uc / 2
    radius_compressive = abs(sigma_uc) / 2
    x_compressive = center_compressive + radius_compressive * np.cos(theta)
    y_compressive = radius_compressive * np.sin(theta)
    
    # Mohr's Circle for Shear
    center_shear = 0
    radius_shear = tau_us
    x_shear = radius_shear * np.cos(theta)
    y_shear = radius_shear * np.sin(theta)
    
    # Mohr's Circle for Specific Stress
    center_specific = specific_sigma / 2
    radius_specific = ((specific_sigma / 2)**2 + (specific_tau)**2)**0.5
    x_specific = center_specific + radius_specific * np.cos(theta)
    y_specific = radius_specific * np.sin(theta)

    # Plot
    plt.figure(figsize=(12, 10))
    
    # Tensile Mohr's Circle
    plt.plot(x_tensile, y_tensile, label='Tensile Mohr\'s Circle (σ_ut)', color='gold')
    
    # Compressive Mohr's Circle
    plt.plot(x_compressive, y_compressive, label='Compressive Mohr\'s Circle (σ_uc)', color='blue')
    
    # Shear Mohr's Circle
    plt.plot(x_shear, y_shear, label='Shear Mohr\'s Circle (τ_us)', color='red')
    
    # Specific Stress Mohr's Circle
    plt.plot(x_specific, y_specific, label='Specific Mohr\'s Circle (σ, τ)', color='green', linestyle='--')
    
    # Plot styling
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Normal Stress σ(kPa)', fontsize=12)
    plt.ylabel('Shear Stress τ(kPa)', fontsize=12)
    plt.title('Mohr\'s Circles for Ultimate Stresses and Specific Stress', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    # Safety check
    for angle in range(0, 360):
        rad = np.deg2rad(angle)
        specific_point_x = center_specific + radius_specific * np.cos(rad)
        specific_point_y = radius_specific * np.sin(rad)

        # Check if the point is outside the tensile, compressive, and shear circles
        outside_tensile = (specific_point_x - center_tensile) ** 2 + specific_point_y ** 2 > radius_tensile ** 2
        outside_compressive = (specific_point_x - center_compressive) ** 2 + specific_point_y ** 2 > radius_compressive ** 2
        outside_shear = specific_point_x ** 2 + specific_point_y ** 2 > radius_shear ** 2

        if outside_tensile and outside_compressive and outside_shear:
            plt.title('Mohrs Circles for Ultimate Stresses and Specific Stress (FAIL)', fontsize=16)
            plt.grid(True)
            plt.legend()
            plt.axis('equal')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('Normal Stress σ (kPa)', fontsize=12)
            plt.ylabel('Shear Stress τ (kPa)', fontsize=12)
            plt.text(0.5 * sigma_ut, 0.5 * tau_us, 'FAIL', color='red', fontsize=30, fontweight='bold', ha='center')
            plt.show()
            print("FAIL: Specific stress is outside of all ultimate stress circles.")
            return

    plt.title('Mohrs Circles for Ultimate Stresses and Specific Stress (SAFE)', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Normal Stress σ (kPa)', fontsize=12)
    plt.ylabel('Shear Stress τ (kPa)', fontsize=12)
    plt.text(0.5 * sigma_ut, 0.5 * tau_us, 'SAFE', color='green', fontsize=30, fontweight='bold', ha='center')
    plt.show()
    print("SAFE: Specific stress is within the ultimate stress circles.")

# Example usage:
# plot_mohrs_circle(sigma_ut, sigma_uc, tau_us, specific_sigma, specific_tau)

# Example usage with user input
length = float(input("Enter the length of the beam (m): "))
width = float(input("Enter the width of the beam (m): "))
height = float(input("Enter the height of the beam (m): "))
beam = SimpleBeam(length=length, width=width, height=height)

while True:
    print("\n1. Add point load")
    print("2. Add uniform load")
    print("3. Add moment load")
    print("4. Delete load")
    print("5. Calculate reactions and plot diagrams")
    print("6. Add crack information")
    print("7. Perform Mohr's Circle analysis and failure diagnostics")
    print("8. Exit")

    try:
        choice = int(input("Choose an option: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    if choice == 1:
        load = -float(input("Enter the magnitude of the point load (negative for downward, kN): "))
        position = float(input("Enter the position of the point load (m): "))
        beam.add_point_load(load, position)
    elif choice == 2:
        load = -float(input("Enter the magnitude of the uniform load (negative for downward, kN/m): "))
        start = float(input("Enter the start position of the uniform load (m): "))
        end = float(input("Enter the end position of the uniform load (m): "))
        beam.add_uniform_load(load, start, end)
    elif choice == 3:
        moment = float(input("Enter the magnitude of the moment load (positive for clockwise, kNm): "))
        position = float(input("Enter the position of the moment load (m): "))
        beam.add_moment_load(moment, position)

    elif choice == 4:
        if not beam.loads:
            print("No loads to delete.")
        else:
            while True:
                try:
                    print("Choose the type of load to delete: \n1. Point load \n2. Uniform load \n3. Moment load")
                    load_type_choice = int(input("Choose an option (enter -1 to go back): "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue
                if load_type_choice == -1:
                    break

                filtered_loads = [
                    (original_index, load) for original_index, load in enumerate(beam.loads)
                    if (load_type_choice == 1 and load['type'] == 'point') or
                      (load_type_choice == 2 and load['type'] == 'uniform') or
                      (load_type_choice == 3 and load['type'] == 'moment')
                ]

                if not filtered_loads:
                    print("No loads of the selected type.")
                    break
                else:
                    print("List of loads available for deletion:")
                    for filtered_index, (original_index, load) in enumerate(filtered_loads):
                        print(f'Load {filtered_index}: {load}')
                    try:
                        filtered_index = int(input("Enter the number of the load to delete (enter -1 to go back): "))
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue

                    if filtered_index == -1:
                        break

                    if 0 <= filtered_index < len(filtered_loads):
                        original_index = filtered_loads[filtered_index][0]
                        beam.delete_load(original_index)
                        print(f"Load ({original_index}) successfully deleted.")

                        filtered_loads = [
                            (original_index, load) for original_index, load in enumerate(beam.loads)
                            if (load_type_choice == 1 and load['type'] == 'point') or
                                (load_type_choice == 2 and load['type'] == 'uniform') or
                                (load_type_choice == 3 and load['type'] == 'moment')
                        ]

                        if not filtered_loads:
                            print("All loads of the selected type have been deleted.")
                            break
                    else:
                        print("Invalid input.")

    elif choice == 5:
        reaction_a, reaction_b = beam.calculate_reactions()
        print(f'Reaction A: {reaction_a} kN')
        print(f'Reaction B: {reaction_b} kN')
        beam.plot_results()
        
        while True:
            x_value = float(input("Enter the position x to calculate shear, moment, and stress (m) (enter -1 to exit): "))
            if x_value < 0:
                break
            y_value = float(input("Enter the y position (distance from the neutral axis in the height direction, m): "))
            z_value = float(input("Enter the z position (distance from the centroid axis in the width direction, m): "))
            shear, moment = beam.get_shear_force_and_moment_at(x_value)
            sigma, tau = beam.calculate_stress(x_value, y_value, z_value)
            stress_tensor = beam.calculate_stress_tensor(x_value, y_value, z_value)
            print(f'Shear force at x = {x_value} m: {shear} kN')
            print(f'Moment at x = {x_value} m: {moment} kNm')
            print(f'Bending stress (\u03C3) at x = {x_value} m: {sigma} kPa')
            print(f'Shear stress (\u03C4) at x = {x_value} m: {tau} kPa')
            print(f'Stress tensor at x = {x_value} m:\n{stress_tensor} (kPa)')

    elif choice == 6:
        num_cracks = int(input("Enter the number of cracks: "))
        for _ in range(num_cracks):
            crack_position = float(input("Enter the position of the crack (m): "))
            crack_height = float(input("Enter the height of the crack (m): "))
            beam.add_crack(crack_position=crack_position, crack_height=crack_height)

        while True:
            x_value = float(input("Enter the position x to calculate stress considering cracks (m) (enter -1 to exit): "))
            if x_value < 0:
                break
            y_value = float(input("Enter the y position (distance from the neutral axis in the height direction, m): "))
            z_value = float(input("Enter the z position (distance from the centroid axis in the width direction, m): "))
            shear, moment = beam.get_shear_force_and_moment_at(x_value)
            applicable_cracks = [crack for crack in beam.cracks if crack['position'] == x_value]
            if applicable_cracks:
                stress_tensor = beam.calculate_stress_tensor(x_value, y_value, z_value)
                print(f'Stress tensor at x = {x_value} m considering cracks: \n{stress_tensor} (kPa)')
            else:
                sigma, tau = beam.calculate_stress(x_value, y_value, z_value, shear, moment)
                stress_tensor = beam.calculate_stress_tensor(x_value, y_value, z_value)
                print(f'Stress tensor at x = {x_value} m without considering cracks: \n{stress_tensor} (kPa)')

    elif choice == 7:
        print("\n1. Perform Mohr's Circle analysis")
        print("2. Perform failure diagnostics")
        sub_choice = int(input("Choose an option: "))
        if sub_choice == 1:
            sigma_ut = float(input("Enter the ultimate tensile stress (\u03C3_ut) (kPa): "))
            sigma_uc = float(input("Enter the ultimate compressive stress (\u03C3_uc) (kPa): "))
            tau_us = float(input("Enter the ultimate shear stress (\u03C4_us) (kPa): "))

            x_value = float(input("Enter the position x for Mohr's Circle analysis (m): "))
            y_value = float(input("Enter the y position (distance from the neutral axis in the height direction, m): "))
            z_value = float(input("Enter the z position (distance from the centroid axis in the width direction, m): "))
            specific_sigma, specific_tau = beam.calculate_stress(x_value, y_value, z_value)
            
            plot_mohrs_circle(sigma_ut, sigma_uc, tau_us, specific_sigma, specific_tau)

        elif sub_choice == 2:
            sigma_y = float(input("Enter the yield stress (\u03C3_y) (kPa): "))
            x_value = float(input("Enter the position x for failure diagnostics (m): "))
            shear, moment = beam.get_shear_force_and_moment_at(x_value)

            applicable_cracks = [crack for crack in beam.cracks if crack['position'] == x_value]
            if applicable_cracks:
                crack = applicable_cracks[-1]
                modified_height = beam.height - crack['height']
            else:
                modified_height = beam.height

            vertical_force_sum = sigma_y * beam.width * modified_height

            yield_moment = vertical_force_sum * modified_height * 0.25

            print(f"Yield moment: {yield_moment} kNm")
            print(f"Beam resisting moment: {moment} kNm")
            if yield_moment > moment:
                print(f"Failure at x = {x_value} m: Yield moment exceeds resisting moment.")
            else:
                print(f"Safe at x = {x_value} m: Resisting moment exceeds yield moment.")

    elif choice == 8:
        break

    else:
        print("Invalid choice. Please try again.")
