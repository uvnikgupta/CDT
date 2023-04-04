# CDT
This repo was created because the original installation steps @ https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html does not work on Ubuntu 22.04

If anyone wants a script that automatically installs all the dependencies and the CDT on Ubuntu 22.04, just clone this repo and run `install_cdt.sh`
<br><br>
<hr>
Below is the process that I followed to get this to work. This is for my own reference for a future me but if someone is also interested to know, feel free to read through.<br><br>
I started off with an Ubuntu 22.04 VM and tried a lot of different ways to install CDT. At the end I could successsfully install it. But then I did not remember the steps I executed. Hence I started the process all over again but this time making sure that was noting down the steps. Even then it took me a couple of iterations till I got the correct sequence that I could put into a script to install it correctly.

I used WSL and am capturing the steps that I followed to get a clean version of Ubuntu 22.04 and restart the installation in each iteration
- Enable WSL on my Windows 10 (ref : https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview)
- Check WSL version using <code> wsl -l -v </code>
- Unregister any previous ubuntu 22.04 installation using <code> wsl --unregister Ubuntu-22.04 </code>

Then follow the below steps
1. From MS store install Ubuntu 22.04.02 LTS
2. From windows command select Ubunto 22.04.02 LTS
3. Enter a username
4. Enter a password
5. Run "sudo apt-get update -y && sudo apt-get install git -y"
6. <code>git clone https://github.com/uvnikgupta/CDT.git </code>
7. <code>chmod 755 CDT/install_cdt.sh && CDT/install_cdt.sh </code>
8. Update the installation script to fix errors
9. commit and push the changes
10. From windows command select Ubuntu --> App settings
11. Select "Reset" in the Ubuntu settings
12. Unregister the ubuntu 22.04 installation using <code> wsl --unregister Ubuntu-22.04 </code>

Repeat step 1 to 12 till I get the installation working without any errors
