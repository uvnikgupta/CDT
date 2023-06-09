start=$(date +%s)

# Check proxy settings and export it in this shell for subsequent subshells to inherit.
if [[ $http_proxy || $HTTP_PROXY ]]; then
  proxy=$http_proxy || $HTTP_PROXY
  substr="http://"
  proxy=${proxy/$substr}
  export http_proxy=http://$proxy
  export https_proxy=https://$proxy
fi

# Check if connection to internet works
wget --timeout=30 http://example.com > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
  echo "ERROR: Cannot reach internet. Check your connection. If you are behind a firewall, check your proxy settings"
  echo "Exiting installation ..."
  exit
fi

sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install software-properties-common -y
sudo apt-get install libharfbuzz-dev libfribidi-dev libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev -y
sudo apt-get install libssl-dev libgmp3-dev git build-essential libv8-dev libcurl4-openssl-dev libgsl-dev libxml2-dev -y

# Install Python 3.10
sudo -E add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt install python3.10 -y
sudo apt install python3-pip python3.10-venv python3.10-distutils python3.10-gdbm python3.10-tk -y

# Install R
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo gpg --dearmor --yes -o /usr/share/keyrings/r-project.gpg
echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | sudo tee -a /etc/apt/sources.list.d/r-project.list
sudo apt-get update -y
sudo apt-get install --no-install-recommends r-base r-base-dev -y

# Insall CDT
cd ~ && git clone http://github.com/FenTechSolutions/CausalDiscoveryToolbox.git
if [[ $proxy ]]; then
  sudo pip3 install --proxy $http_proxy cdt
  cd CausalDiscoveryToolbox && sudo pip3 install --proxy $http_proxy -r requirements.txt && sudo python3 setup.py install develop --user  && cd ~

  # Modify the Renviron.site file for proxy. Used by R to install it's packages
  sudo cp /usr/lib/R/etc/Renviron.site /usr/lib/R/etc/Renviron.site.orig
  sudo sh -c 'echo >> /usr/lib/R/etc/Renviron.site'
  sudo -E sh -c 'echo http_proxy=$http_proxy >> /usr/lib/R/etc/Renviron.site'
  sudo -E sh -c 'echo https_proxy=$http_proxy >> /usr/lib/R/etc/Renviron.site'
else
  sudo pip3 install cdt
  cd CausalDiscoveryToolbox && sudo pip3 install -r requirements.txt && sudo python3 setup.py install develop --user && cd ~
fi

#Install R libraries
sudo -i R -q --no-save <<'EOF'
install.packages("V8")
install.packages("sfsmisc")
install.packages("clue")
install.packages("lattice")
install.packages("devtools")
install.packages("MASS")
install.packages("BiocManager")
install.packages("igraph")
install.packages("discretecdAlgorithm")

install.packages("https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz", repos=NULL, type="source")
install.packages("https://cran.r-project.org/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz", repos=NULL, type="source")

BiocManager::install(c("bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"), force=TRUE, update = TRUE, ask = FALSE)
install.packages("https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz", repos=NULL, type="source")
install.packages("https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz", repos=NULL, type="source")
install.packages("https://cran.r-project.org/src/contrib/sparsebnUtils_0.0.8.tar.gz", repos=NULL, type="source")
install.packages("https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz", repos=NULL, type="source")
install.packages("https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz", repos=NULL, type="source")

library(devtools); install_github("cran/CAM"); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT")
EOF

end=$(date +%s)
runtime=$(expr $end - $start)
echo "Total installation time: $runtime seconds"
