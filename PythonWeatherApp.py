import requests

url = "https://weatherapi-com.p.rapidapi.com/current.json"
'''
Query parameter based on which data is sent back. It could be following: Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508 
city name e.g.: q=Paris US zip e.g.: q=10001 UK postcode e.g: 
q=SW1 Canada postal code e.g: q=G2J metar: e.g: q=metar:EGLL iata:<3 digit airport code> e.g: q=iata:DXB 
auto:ip IP lookup e.g: q=auto:ip IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
'''
querystring = {"q":"New Delhi,INDIA"}

headers = {
	"X-RapidAPI-Key": "3ecf41067bmsh65f8d8de3a02231p1a82efjsnd732f8f008fb",
	"X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())
print(response.json())
