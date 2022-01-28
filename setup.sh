mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

tar xvzf scip-x.y.z.tgz                                                       # unpack the tarball
cd scip-x.y.z                                                                 # change into the directory
mkdir build                                                                   # create a new directory
cd build                                                                      # change directories
cmake .. -DCMAKE_INSTALL_PREFIX=<install/dir> [-DSOPLEX_DIR=/path/to/soplex]  # configure the build
make                                                                          # start compiling SCIP
make check                                                                    # (recommended) check build
make install                                                                  # (optional) install SCIP executable, library, and headers
