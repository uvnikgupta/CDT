# CDT

This repo was created because the original installation steps @ https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html do not work on Ubuntu 22.04 :woozy_face:

I extended this further and created installation for Windows also. So all of you Windows fans out there can also benefit from this

Mac fans, soorrry, you need to do this on your own. But, I guess, you can get some inspriation from here.

The installation script, `install_cdt.sh`, for ubuntu is under `installation\ubuntu_22.04\`and `install_cdt.ps1`, for windows is under `installation\windows_10\`. These scripts will automagically install CDT and all it's dependencies 💃. But there are a few (actually just 3) prerequisties that you need to take care. These will ensure that you enjoy your ☕️ while your environment is getting created. 

**For Ubuntu:**

1. Ubuntu 22.04.02 LTS
2. sudo privlidges
3. wget installed

**For Windows:**

1. Windows 10
2. Anaconda installed
3. Run the Powershell with admin privlidges

To check if the installation was successfull you can run either one or both of the python scripts avaialble in this repo<br>
<code>python3 CDT/cdt_basic.py</code><br>
<code>python3 CDT/cdt_advanced.py</code><br><br>
<code>cdt_advanced.py</code> will run much faster if you have GPUs, else it will easily take a while depending upon your CPU and the load on it. When I ran it on my Windows PC without a GPU, while doing other stuff in parallel, it took almost a day to complete.
<br>

#### Issues you may face :face_with_head_bandage: :

1. **Timeout while installing an ubuntu or python library** :hourglass_flowing_sand: --> Check if you have provided the correct proxy
2. **Timeout while installing R library** :hourglass_flowing_sand: --> Some systems may have the sudoers configuration set to reset the environment variables when running sudo. In this case, the -E option may not work and you may need to modify the sudoers configuration to allow the http_proxy environment variable to be preserved<br><br>

<hr>
Below is the process that I followed to get this to work on ubuntu. This is for my reference for a future me but if you are also interested to know about the grind, feel free to read through.:nerd_face:<br><br>
I started off with an Ubuntu 22.04.02 LTS VM and tried a lot of different ways to install CDT. At the end I could successsfully install it. But then I did not remember the steps I had executed to get this to work :facepalm:. Hence I started the process all over again. This time, making sure that I was noting down all the steps I executed. Even then it took me more than a couple of iterations till I could get the correct versions, commands and the sequence in the right order that would make it run without any errors.<br><br>


I used WSL and am capturing the steps that I followed to start each iteration of installation process with a a clean version of Ubuntu 22.04

- Enable WSL on my Windows 10 (ref : https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview)
- Check WSL version using <code> wsl -l -v </code>
- Unregister any previous ubuntu 22.04 installation using <code> wsl --unregister Ubuntu-22.04 </code>

Then follow the below steps

1. From MS store install Ubuntu 22.04.02 LTS
2. From windows command select Ubunto 22.04.02 LTS
3. Enter a username
4. Enter a password
5. <code>sudo apt-get update -y && sudo apt-get install git -y </code>
6. <code>git clone https://github.com/uvnikgupta/CDT.git </code>
7. <code>chmod 700 CDT/installation/ubuntu_22.04/install_cdt.sh && CDT/installation/ubuntu_22.04/install_cdt.sh </code>
8. Update the installation script to fix errors
9. commit and push the changes
10. From windows command select Ubuntu --> App settings
11. Select "Reset" in the Ubuntu settings
12. Unregister the ubuntu 22.04 installation using <code> wsl --unregister Ubuntu-22.04 </code>

:repeat:Repeat step 1 to 12 till I get the installation working without any errors
<br><br><br>
On Windows also I followed a similar process but since I did not have the luxury of creating Windows VMs from scratch for each iteration, I managed by uninstallting and cleaning the environment using the following steps:

1. Uninstall R using `winget rm --id RProject.R`
2. Uninstall the conda env using `conda env remove -n <name of the env>`
3. Manually delete the conda env folder
