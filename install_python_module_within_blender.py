
import sys
import subprocess
import os
import platform
import bpy

def isWindows():
    return os.name == 'nt'

def isMacOS():
    return os.name == 'posix' and platform.system() == "Darwin"

def isLinux():
    return os.name == 'posix' and platform.system() == "Linux"

def python_exec():
    
    if isWindows():
        return os.path.join(sys.prefix, 'bin', 'python.exe')
    elif isMacOS():
    
        try:
            # 2.92 and older
            path = bpy.app.binary_path_python
        except AttributeError:
            # 2.93 and later
            import sys
            path = sys.executable
        return os.path.abspath(path)
    elif isLinux():
        import sys
        #print(sys.prefix)
        #return os.path.join(sys.prefix, 'sys.prefix/bin', 'python')
        #return os.path.join(sys.prefix, 'bin', 'python3.7m')
        #return os.path.join(sys.prefix, 'bin', 'python3.9')
        return os.path.join(sys.prefix, 'bin', 'python3.10')
    else:
        print("sorry, still not implemented for ", os.name, " - ", platform.system)



def installModule(packageName=None, uninstall=False, show_list=False):
    
    try:
        subprocess.call([python_exe, "import ", packageName])
    except Exception as e:
        print("Exception message:", e)
        python_exe = python_exec()
        print(python_exe)
        
        if show_list:
            subprocess.call([python_exe, "-m", "pip", "freeze"])
        if packageName is None:
            return
        
        # upgrade pip
        subprocess.call([python_exe, "-m", "ensurepip"])
        subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        
        if uninstall:
            subprocess.call([python_exe, "-m", "pip", "uninstall", packageName])    
        else:
            # install required packages
            subprocess.call([python_exe, "-m", "pip", "install", packageName])
            #subprocess.call(["pip", "install", packageName])
            

        
#installModule("PyYAML")
#installModule("opencv-python")
#installModule("scipy")
installModule(show_list=True)


'''
Packages used during development (a lot were already installed in the pre-built Blender)

autopep8==1.6.0
certifi==2021.10.8
charset-normalizer==2.0.10
contourpy==1.0.7
cycler==0.11.0
Cython==0.29.30
fonttools==4.39.3
idna==3.3
kiwisolver==1.4.4
lxml==4.9.3
numpy==1.23.5
opencv-python==4.8.0.74
packaging==23.0
pycodestyle==2.8.0
PyYAML==6.0
requests==2.27.1
scipy==1.11.1
toml==0.10.2
urdf-parser-py==0.0.4
urllib3==1.26.8
zstandard==0.16.0
'''