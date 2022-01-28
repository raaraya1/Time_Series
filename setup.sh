mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

apt-get update

apt-get install build-essential libreadline-dev libz-dev libgmp3-dev lib32ncurses5-dev libboost-program-options-dev libblas-dev

tar xvzf scipoptsuite-7.0.2.tgz
cd scipoptsuite-7.0.2

apt-get install make
make

make gcg
