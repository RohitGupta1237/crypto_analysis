# Basic Crypto Analysis
Some basic visualizations and analysis for crypto.

### Linux
````
sudo apt install python3-pip git
pip install pandas-datareader mplfinance seaborn
git clone https://github.com/Am0rphous/crypto-analysis
cd crypto-analysis
python3 crypto_correlation.py 
python3 crypto_candle.py 
````

### MacOS
- Install the package manager [Homebrew](https://brew.sh/)
- Then run:
````
brew install python3 git
pip3 install pandas-datareader mplfinance seaborn
git clone https://github.com/Am0rphous/crypto-analysis
cd crypto-analysis
python3 crypto_correlation.py 
python3 crypto_candle.py 
````

### Preview
Running `python3 crypto_correlation.py` will create the first two images.

![crypto_correlation_1.png](images/crypto_correlation_1.png?raw=true)
![crypto_correlation_2.png](images/crypto_correlation_2.png?raw=true)

Running `python3 crypto_candle.py` will create this image.

![crypto_candle.png](images/crypto_candle.png?raw=true)



