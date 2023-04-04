# CDT
This repo was created because the original installation steps @ https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html do not work on Ubuntu 22.04 :woozy_face:

If anyone wants a script that automatically installs all the dependencies and the CDT on Ubuntu 22.04, just clone this repo and run `install_cdt.sh`<br><br>
To check if the installation was successfull you can run either one or both of the python scripts avaialble in this repo<br>
<code>python3 CDT/cdt_basic.py</code><br>
<code>python3 CDT/cdt_advanced.py</code><br><br>
<code>cdt_advanced.py</code> will run much faster if you have GPUs, else it will easily take a while depending upon your CPU and the load on it. When I ran it on my PC without a GPU, while doing other stuff in parallel, it took almost half a day to complete.
<br>
#### Issues you may face :face_with_head_bandage: :
1. **Timeout while installing an ubuntu or python library** :hourglass_flowing_sand: --> Check if you have provided the correct proxy
2. **Timeout while installing R library** :hourglass_flowing_sand: --> Some systems may have the sudoers configuration set to reset the environment variables when running sudo. In this case, the -E option may not work and you may need to modify the sudoers configuration to allow the http_proxy environment variable to be preserved<br><br>
<hr>
Below is the process that I followed to get this to work. This is for my reference for a future me but if you are also interested to know about the grind, feel free to read through.:nerd_face:<br><br>
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
7. <code>chmod 700 CDT/install_cdt.sh && CDT/install_cdt.sh </code>
8. Update the installation script to fix errors
9. commit and push the changes
10. From windows command select Ubuntu --> App settings
11. Select "Reset" in the Ubuntu settings
12. Unregister the ubuntu 22.04 installation using <code> wsl --unregister Ubuntu-22.04 </code>

:repeat:Repeat step 1 to 12 till I get the installation working without any errors
