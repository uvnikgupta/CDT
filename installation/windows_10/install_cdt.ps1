do {
    Write-Host "Enter the conda env name for cdt installation: " -ForegroundColor Green -NoNewline
    $condaenv = Read-Host
} while (-not $condaenv -ne "")

# Install R
winget install --disable-interactivity --accept-package-agreements --accept-source-agreements -e --id  RProject.R

# Get the installation path for R
$rpath = Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\* | Where-Object {$_.DisplayName -like "R*"} | Select-Object -ExpandProperty InstallLocation | Select-String -Pattern "R\\R-" | Select-Object -ExpandProperty Line
$rpath = $rpath + "bin"

# check is the installation path exists
if (Test-Path $rpath -PathType Container) {
    # append the R installation path to systme path
    $env:Path = "$env:Path;$rpath"
} else {
    throw "Oops!! Could not get the installation path for R."
}

# Install RTools
winget install --disable-interactivity --accept-package-agreements --accept-source-agreements -e --id RProject.Rtools -v 4.2

# create conda environment, install cdt and dependent libraries for it to work without errors
conda create -y -n $condaenv python=3.10
conda activate $condaenv
conda install -y -c conda-forge cdt
conda install -y networkx=2.7
conda deactivate

# install R libraries
RScript.exe  -e "install.packages('igraph', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('V8', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('sfsmisc', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('clue', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('lattice', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('devtools', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('MASS', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('igraph', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('discretecdAlgorithm', repos='http://cran.us.r-project.org')"
RScript.exe  -e "install.packages('randomForest', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('fastICA', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('bnlearn', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('pcalg', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('kpcalg', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('glmnet', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('mboost', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('iterators', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('vctrs', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('fastmap', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('htmltools', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('promises', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('stringr', repos='https://cran.r-project.org')"
RScript.exe  -e "install.packages('https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz', repos=NULL, type='source')"
RScript.exe  -e "install.packages('https://cran.r-project.org/src/contrib/sparsebnUtils_0.0.8.tar.gz', repos=NULL, type='source')"
RScript.exe  -e "install.packages('https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz', repos=NULL, type='source')"
RScript.exe  -e "install.packages('https://cran.r-project.org/src/contrib/Archive/ccdrAlgorithm/ccdrAlgorithm_0.0.6.tar.gz', repos=NULL, type='source')"
RScript.exe  -e "install.packages('https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz', repos=NULL, type='source')"
RScript.exe  -e "library(devtools); install_github('cran/CAM'); install_github('cran/momentchi2'); install_github('Diviyan-Kalainathan/RCIT')"

# get the conda cdt env path
$cdtpath = conda env list | Select-String -Pattern $condaenv | Select-Object -ExpandProperty Line
$cdtpath = $cdtpath.Split(" ")[-1]  + "\Lib\site-packages\cdt"

$scriptPath = $PSScriptRoot
$paths = "\data\resources", "\utils\R_templates", "\causality\graph\R_templates"

# copy the missing files
foreach ($path in $paths) {
    $sourcepath = $scriptPath + "\missing_in_windows\" + $path
    $destpath = $cdtpath + $path
    Copy-Item -Path $sourcepath -Destination $destpath -Recurse
}

conda activate $condaenv

## Delete R and conda env using the following commands:
# winget rm --id RProject.R; winget rm --id RProject.Rtools
# conda deactivate; conda env remove -n cdt