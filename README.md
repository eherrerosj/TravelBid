# TravelBid candidates API
## FITUR2018 hackathon - Team 10

### Usage to start API listener (Flask server):

#### Locally:
`make serve`

#### From a Docker container:
`make run`


### Usage of API requests:

#### Method 1: candidates
Call `/candidates`

Parameters:
- HotelID: mandatory, numeric
- limit: optional, defaults to 5

Example:
`curl -i -X GET -H "Accept: application/json" 'http://127.0.0.1:5000/candidates?HotelID=1901731&limit=3'`

Response: json list of objects sorted by similarity of each hotel
- recommended_hotel_id: HotelID of the recommended hotel
- HotelName: name of the recommended_hotel_id
- distance_to_input_hotel: distance between the hotel requested and the recommended_hotel_id
`[{"recommended_hotel_id":1825055,"HotelName":"Pension Avantiss","distance_to_input_hotel":1.9482062605},{"recommended_hotel_id":2543167,"HotelName":"Espanya Suite II Gran Via City Center","distance_to_input_hotel":1.3474036181},{"recommended_hotel_id":1971570,"HotelName":"Centro De Madrid","distance_to_input_hotel":1.9373597679}]`
