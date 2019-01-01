import requests

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.data'
birthdata_url = 'https://github.com/mahbubkabir/GA---Data-Science/blob/master/Week%206/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(')') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
print(len(birth_data))
print(len(birth_data[0]))