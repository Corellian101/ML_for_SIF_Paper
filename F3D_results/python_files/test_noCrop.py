import sys
import Raju_Newman_new
sys.path.append("/uufs/chpc.utah.edu/common/home/u0823391/lib")
import PyF3D,os,shutil
import numpy as np
import calcq as cq
import math
import create_plate
#import matplotlib.pyplot as plt

restart = True
a_t = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8])
a_c = np.array([0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
a_c = np.array([0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65])
c_b = 0.2
t = 0.75
l = 24
h= 24
A = a_t*t

f3d = PyF3D.F3DApp()

# Makes Directory for size of Cube
sizeDir = str(l)+'x'+str(h)+'x'+str(round(t,3))
if os.path.exists(sizeDir):
    if not restart:
        shutil.rmtree(sizeDir)
        os.mkdir(sizeDir)
else:
    os.mkdir(sizeDir)
path = os.getcwd()

print >> sys.__stdout__, 'starting'
H = np.linspace(1.5,3.0,5)
H = [12,14,16,18]
H = [0.01]
for a in A:
    C = a/a_c
    for c in C:
        for h in H:
            #model params
            b = c/c_b
            x_cut = b*0.7
            y_cut = x_cut*0.4
            seedloc = x_cut*0.02
            seedglo = b*0.1

            #crack params
            rad = 0.05#h#0.005
            progression_ratio = 1
            num_rings = 3
            num_circ = 12#8 #must be even
            max_aspect_ratio = 2

            #refinement params
            surf_refine = 1.2 
            vol_refine = 2.6
            method = 'FRANC3D'
            
            print(b)
            print >> sys.__stdout__, b
            print >> sys.__stdout__, c
            print(c)
            if x_cut >= b:
                continue
            if y_cut >= b:
                continue
            if c >= x_cut:
                continue
            # Make Directory for a/c and a/t ratio
            dir = 'ac'+str(round(a/c,2))+'_at'+str(round(a/t,2))+'_cb'+str(round(c/b,2))+'_rad'+str(round(rad,3))+'_'+method+'_num_circ'+str(num_circ)

            # Filenames
            jobName = 'equal_bcs_fixed'
            jobName = jobName.replace('.','')
            inp_file = jobName+'.inp'
            sif_file='Calc_SIF.sif'
            #fbd_file = fbd_file.replace('.','')
            #sif_file = sif_file.replace('.','')
            dir = dir.replace('.','')

            # Make directory to where I want data stored
            if os.path.exists(sizeDir+'/'+dir):
                if not restart:
                    shutil.rmtree(sizeDir+'/'+dir)
                else:
                    continue
            os.mkdir(sizeDir+'/'+dir)
            print('DIRECTORY CREATED:  '+sizeDir+'/'+dir)

            # Moves essential files to working Directory
            #shutil.copyfile(inp_file , sizeDir+'/'+dir+'/'+inp_file)
            #shutil.copyfile(jobName+'_RETAINED_ELEMS.txt', sizeDir+'/'+dir+'/'+jobName+'_RETAINED_ELEMS.txt')
            #shutil.copyfile('local_model_4_RETAINED.txt', sizeDir+'/'+dir+'/'+'local_model_RETAINED.txt')
            print('FILES COPIED')

            # Change working directory to where I want data stored
            os.chdir(sizeDir+'/'+dir)
            print('DIRECTORY CHANGED')
            print('Creating inp file')
            create_plate.create_model(b)
            create_plate.create_global(x_cut,y_cut,seedglo)
            local_nodes = create_plate.create_local(b,x_cut,y_cut,seedloc)

            print('Loading inp into FRANC3D')
            try:
                f3d.OpenMeshModel(
                    model_type="ABAQUS",
                    file_name='local_model.inp',
                    global_name='global_model.inp',
                    retained_nodes = local_nodes)
                    #retained_nodes_file='local_model_RETAINED.txt')
            except:
                continue
            f3d.SetMeshingParameters(
                    max_gen_elems=600000,
                    do_surface_refinement=True,
                    volume_meshing_method=method,
                    surf_refine_bdry_factor=surf_refine,
                    volume_refine_factor = vol_refine)

            
            print('Inserting crack')
            try:
                f3d.InsertParamFlaw(
                    flaw_type="CRACK",
                    crack_type="ELLIPSE",
                    flaw_params=[c,a],
                    rotation_axes=[1],
                    rotation_mag=[90],
                    translation=[0,0,t],
                    radius=rad,
                    progression_ratio=progression_ratio,
                    num_rings=num_rings,
                    num_circ=num_circ,
                    max_aspect_ratio=max_aspect_ratio)
            except:
                continue

            print('Running Abaqus analysis')

            try:
                f3d.RunAnalysis(
                    model_type="ABAQUS",
                    file_name='Cracked.fdb',
                    flags=["NO_WRITE_TEMP","TRANSFER_BC","NO_CFACE_TRACT","NO_CFACE_CNTCT"],
                    merge_tol=0.0001,
                    connection_type="CONSTRAIN",
                    executable='abaqus',
                    command='abaqus job=Cracked_full cpus=6 -interactive -analysis',
                    global_model='global_model.inp',
                    merge_surf_labels=["Part-1-1_cut_local"],
                    global_surf_labels=["Part-1-1_cut_global"],
                    locglob_constraint_adjust=False,
                    locglob_constraint_pos_tol=0.001)
            except:
                continue

            print('computing sif')
            f3d.ComputeSif(
                    sif_method = 'M_INTEGRAL')
            f3d.WriteSif(
                file_name=sif_file,
                crack_step=0,
                load_step=1,
                flags=["TAB","KI","KII","KIII","CRD"])

            f3d.ComputeSif(
                    sif_method = 'DISP_CORR')
            f3d.WriteSif(
                file_name='d_corr.sif',
                crack_step=0,
                load_step=1,
                flags=["TAB","KI","KII","KIII","CRD"])


            # READ SIF DATA AND PLOT IT
            shutil.copy('Cracked_full.dtp','../../dtp_files/'+dir+'.dtp')
            data = np.genfromtxt(sif_file)
            data_d = np.genfromtxt('d_corr.sif')
            K=data[:,1]
            Ncoord=data[:,0]
            x = data[:,4]

            K_d = data_d[:,1]
            x_d = data_d[:,0]
            #phi = np.arccos(x/c)
            #phi_norm = 2*phi/math.pi

            phi = np.linspace(0,math.pi/2,len(x))
            
            S=400
            Q=cq.empirical_shape_factor(a,c)
            F_data=K/(S*np.sqrt(math.pi*a/Q))
            F_d=K_d/(S*np.sqrt(math.pi*a/Q))
            print([a/c,a/t])
            RN_data = Raju_Newman_new.F_s(a/c,a/t,0,phi)
            F_datatable = np.column_stack((Ncoord, F_data, phi/math.pi, RN_data,x_d,F_d,x))
            np.savetxt('/uufs/chpc.utah.edu/common/home/u1155988/data_for_bingo/'+dir+'.csv',F_datatable)
            #plt.figure()
            #plt.plot(df['NCoord'], df['KI'])
            #plt.xlabel('Distance along crack Face (Normalized)')
            #plt.ylabel(r'SIF - $K_I$')
            #plt.title('a/c = '+str(round(i/j,2))+' a/t = '+str(i))
            #plt.savefig(dir+'.png', bbox_inches = 'tight')
            os.chdir(path)
