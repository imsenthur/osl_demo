import os
import numpy as np
import time
from signal import signal, SIGINT
from flexsea import flexsea as flex
from flexsea import fxUtils as fxu  # pylint: disable=no-name-in-module
from flexsea import fxEnums as fxe  # pylint: disable=no-name-in-module
from DataLogger import dataLogger

# from walk_functions import *

def ME461_demo():

    # initilize and stream Actpack 

    port_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ports.yaml")
    ports, baud_rate = fxu.load_ports_from_file(port_cfg_path)
    fxs = flex.FlexSEA()

    dev_id_0 = fxs.open(ports[0], baud_rate,6)
    fxs.start_streaming(dev_id_0, 500, log_en=False)


   ######################## SET GAINS ######################################

    # Gains are, in order: kp, ki, kd, K, B & ff
    # fxs.set_gains(dev_id_0, 50, 200, 0, 100, 20, 80) 
    
    kp_pos = 300
    ki_pos = 20
    kd_pos = 40
    kff_pos = 128

    kp_cur = 50
    ki_cur = 200
    kff_cur = 80

    stiffness = 60
    damping = 100

    ########################################################################

    time.sleep(1)

    data_knee = fxs.read_device(dev_id_0)
    initial_mot_ang = data_knee.mot_ang

    ######################## SET CURRENT THRESHOLD ########################

    current_thresh = 10000
    current_max = 40000

    ############################ SET DURATION ##############################

    duration = 10 # (sec) -- Shouldn't be less than 5 seconds
    
    ########################################################################

    t_step = 0.002 # (sec)
    freq = 1/(t_step) # frequency
    t = np.linspace(0, duration, int(duration*freq)) # creating a time array
    
    deg2counts = (2.0**14)/360.0 # joint degree to motor count
    counts2deg = 1/deg2counts
    transmission_ratio = 4.611*9 # OSL * Dephy internal transmision ratio

    ref_type = input('What type of reference would you like: \n 1- Sine wave \n 2- Step \n 3- Hold\n')
    cont_type = input('What type of control would you like: \n 1- Position control \n 2- Impedance control \n 3- Current control\n')

    if ref_type == '1': # Sine wave option
        if cont_type == '3':
            A = float(input('What is the amplitude of the sine wave (mA): '))
            F = float(input('What is the frequency of the sine wave (Hz)?: '))
            ref = A*np.sin(t*F*2*3.14) # generating a current ref
            
        else:
            A = float(input('What is the amplitude of the sine wave (deg): '))
            F = float(input('What is the frequency of the sine wave (Hz)?: '))
            ref = initial_mot_ang + A*np.sin(t*F*2*3.14)*transmission_ratio*deg2counts # generating a position ref
        
        print('The sine wave will be centered in the range of motion')

        
        

    if ref_type == '2': # Step option
    


        if cont_type == '3':

            S = float(input('How large would you like the step (mA)?: '))
            refA = np.zeros(int(len(t)*0.3)) 
            refB = S*np.ones(len(t) - int(len(t)*0.3))
            ref =  np.append(refA, refB) # generating a current ref

        else:

            S = float(input('How large would you like the step (deg)?: '))
            refA = np.zeros(int(len(t)*0.3)) 
            refB = S*np.ones(len(t) - int(len(t)*0.3))
            ref = initial_mot_ang + np.append(refA, refB)*deg2counts*transmission_ratio # generating a position ref

        print('The step will occur after X seconds')


    if ref_type == '3':

        if cont_type == '3':

            H = float(input('What value would you like to hold (mA)?:'))
            ref = H*np.ones(len(t)) # generating a current ref

        else:

            H = float(input('What value would you like to hold (deg)?:'))
            ref = initial_mot_ang + H*np.ones(len(t))*deg2counts*transmission_ratio # generating a position ref

    trial_num = int(input('Trial Number? '))
    filename = 'Test%i_ME461' % trial_num
    dl = dataLogger(filename + '.csv')

    counter = 0
    k = 0.0

    t0 = time.time()

    for i in ref: # loop over time duration 

        data_knee = fxs.read_device(dev_id_0)
        motor_current = abs(data_knee.mot_cur)
        # print('motor current', motor_current/1000,'A')

        if motor_current > current_max:

            # When we exit we want the motor to be off
            for w in range(30):
                fxs.send_motor_command(dev_id_0, fxe.FX_NONE, 0)
                
            print('high current', motor_current/1000,'A') 
            time.sleep(0.5)
            break 


        time.sleep(t_step - .00057)
        motor_angle = data_knee.mot_ang
        i = i.item() # convert np.float64 type to float type
        i = int(i)   # convert float type to int type 
        friction_delay = 0.2 #delays current safety switch
        # holds the current protection logic for 0.05 sec to give overcome static friction     
        if (freq * friction_delay < counter < len(t)*0.3) or (len(t)*0.3 + freq * friction_delay < counter):

            if motor_current > current_thresh: # current protection logic 

                # When we exit we want the motor to be off
                for w in range(30):
                    fxs.send_motor_command(dev_id_0, fxe.FX_NONE, 0)
                    
                print('high current', motor_current/1000,'A') 
                time.sleep(0.5)
                break 


        if cont_type == '1': # position control option

            fxs.send_motor_command(dev_id_0, fxe.FX_POSITION, i)
            fxs.set_gains(dev_id_0, kp_pos, ki_pos, kd_pos, 0, 0, kff_pos) 

            
        if cont_type == '2': # impedance control option
        
            fxs.send_motor_command(dev_id_0, fxe.FX_IMPEDANCE, i)
            fxs.set_gains(dev_id_0, kp_cur, ki_cur, 0, stiffness, damping, kff_cur) 

       
        if cont_type == '3': # current control option
            fxs.send_motor_command(dev_id_0, fxe.FX_CURRENT, i)
            fxs.set_gains(dev_id_0, kp_cur, ki_cur, 0, 0, 0, kff_cur) 

        # prep data for appending & printing
        data = [t[counter]] + [i] + [motor_angle] + [data_knee.mot_cur] + [initial_mot_ang] + [cont_type] + [data_knee.mot_volt] + [data_knee.batt_curr] + [data_knee.batt_volt] + [data_knee.temperature]

        if counter % 10 == 0:

            print('Motor encoder: %.2f Deg;   Motor current : %.2f A; ' % (motor_angle*counts2deg, data_knee.mot_cur/1000))
        dl.appendData(data) 

        counter += 1

            
    for w in range(30):
        # When we exit we want the motor to be off 
        fxs.send_motor_command(dev_id_0, fxe.FX_NONE, 0)
    
    time.sleep(0.5)
    fxs.close(dev_id_0)
    dl.writeOut()

    print ("Iterations: %.2f " % (counter))
    print ("Elapsed Time: %.2f " % (time.time()-t0))
    print ("Mean Frequency: %.2f " % ((counter + 0.0)/(time.time()-t0)))


if __name__ == "__main__":
	ME461_demo()