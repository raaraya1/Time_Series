mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

tar xvzf scip-7.0.2.tgz                                                       # unpack the tarball
cd scip-7.0.2                                                                 # change into the directory
mkdir build                                                                   # create a new directory
cd build                                                                      # change directories
cmake .. -DCMAKE_INSTALL_PREFIX=<streamlit> [-DSOPLEX_DIR=/path/to/soplex]  # configure the build
make                                                                          # start compiling SCIP
make check                                                                    # (recommended) check build
make install                                                                  # (optional) install SCIP executable, library, and headers
