
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import joblib
import os



# Initialize the Flask app
app = Flask(__name__)

model = joblib.load('model_2.pkl')

with open('scaler_2_category.pkl', 'rb') as scaler_file:
    scaler_f = pickle.load(scaler_file)

with open('scaler_2_price.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



# Define available options for dropdowns (same as before)
company_options = ['MG', 'Hyundai', 'Nissan', 'TATA', 'Volkswagen', 'Mahindra', 'Maruti Suzuki', 'Renault', 'Toyota', 'Honda', 'Kia']
fuel_options = ['Diesel', 'Petrol']
tyre_options = ['Needs Replacement', 'New', 'Used']
owner_options = ['First', 'Second', 'Third', 'Fourth']
registration_options = ['DL-XX-XX-XXXX', 'AP-XX-XX-XXXX', 'MH-XX-XX-XXXX', 'KA-XX-XX-XXXX', 'TN-XX-XX-XXXX', 'TS-XX-XX-XXXX', 'JS-XX-XX-XXXX', 'MP-XX-XX-XXXX', 'KL-XX-XX-XXXX', 'PB-XX-XX-XXXX']
transmission_options = ['Automatic', 'Automatic (Tiptronic)', 'Manual', 'unknown_transmission']
certificate_options = ['Not Available', 'Available']
car_name_options = ['Astor', 'Gloster', 'Etios', 'Carens', 'XUV300', 'Eeco', 'Jazz', 'Hector Plus', 'Thar', 'Creta', 'Sunny', 'Vento', 'i10', 'Venue', 'Camry', 'Civic', 'Altroz', 'Micra', 'Corolla', 'Punch', 'Bolero', 'Hector', 'Accord', 'Ignis', 'Verna', 'Land Cruiser', 'Tiago', 'Kwid', 'KUV100', 'Tiguan', 'Ioniq', 'Safari', 'Carnival', 'Sonet', 'Triber', 'Ciaz', 'Marazzo', 'Kicks', 'Nexon', 'Glanza', 'Dzire', 'Innova Crysta', 'T-Roc', 'Polo', 'Yaris', 'Kiger', 'Exter', 'Omni', 'City', 'X-Trail', 'Brezza', 'i20', 'Celerio', 'Harrier', 'Fortuner', 'Taigun', 'XUV700', 'Scorpio', 'WR-V', 'Urban Cruiser', 'Aura', 'Baleno', 'Magnite', 'Alto', 'Swift', 'Innova', 'Alcazar', 'Amaze', 'S-Presso', 'Tigor', 'Santro', 'Seltos', 'CR-V', 'Ertiga']

# Define accessories weights (same as before)
accessory_weights = {
    'Music System': 10,
    'Sunroof': 20,
    'Alloy Wheels': 30,
    'GPS': 40,
    'Leather Seats': 50
}

# Define the route for the homepage3
@app.route('/')
def home():
    return render_template('index.html', 
                           companies=company_options, 
                           fuels=fuel_options, 
                           tyres=tyre_options, 
                           owners=owner_options, 
                           registrations=registration_options,
                           transmissions=transmission_options,
                           certificates=certificate_options,
                           car_names=car_name_options,
                           accessories=accessory_weights)



@app.route('/predict_again', methods=['POST'])
def predict_again():
    return render_template('index.html', 
                           companies=company_options, 
                           fuels=fuel_options, 
                           tyres=tyre_options, 
                           owners=owner_options, 
                           registrations=registration_options,
                           transmissions=transmission_options,
                           certificates=certificate_options,
                           car_names=car_name_options,
                           accessories=accessory_weights)


# Define the route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the data from the form
    company = request.form['company']
    fuel = request.form['fuel']
    tyre = request.form['tyre']
    owner = request.form['owner']
    registration = request.form['registration']
    transmission = request.form['transmission']
    certificate = request.form['certificate']
    car_name = request.form['car_name']
    make_year = int(request.form['make_year'])
    mileage = float(request.form['mileage'])
    

    # Handle accessories
    selected_accessories = request.form.getlist('accessories')
    Sunroof = Alloy_Wheels = GPS = Leather_Seats = Music_System =  0
    for accessory in selected_accessories:
        if accessory == 'Sunroof':
            Sunroof = 1
        if accessory == 'Alloy Wheels':
            Alloy_Wheels = 1
        if accessory == 'GPS':
            GPS = 1
        if accessory == 'Leather Seats':
            Leather_Seats = 1
        if accessory == 'Music System':
            Music_System = 1
    
    def predict_car_price(make_year, mileage, company_name, car_name, fuel_type, tyre_condition,
                        owner_type, registration_number, transmission_type, registration_certificate,Music_System,Sunroof,Alloy_Wheels,GPS,Leather_Seats):
        # Create a DataFrame for input data

        numerical_columns = [make_year,mileage]
        input_data = pd.DataFrame({
            'Company Name': [company_name],
            'Car Name': [car_name],
            'Fuel Type': [fuel_type],
            'Tyre Condition': [tyre_condition],
            'Make Year': [make_year],
            'Owner Type': [owner_type],
            'Registration Number': [registration_number],
            'Mileage': [mileage],
            'Transmission Type': [transmission_type],
            'Registration Certificate': [registration_certificate],
            'Music System':[Music_System],
            'Sunroof':[Sunroof],
            'Alloy Wheels':[Alloy_Wheels],
            'GPS':[GPS],
            'Leather Seats':[Leather_Seats]
        })

        # Print the input data structure for debugging
        print("Input Data Before Preprocessing:")
        print(input_data)

        # Scaling only the relevant numerical columns
        numerical_columns = ['Make Year','Mileage']
        input_data[numerical_columns] = scaler_f.transform(input_data[numerical_columns])

        # Print the input data after scaling for debugging
        print("Input Data After Scaling:")
        print(input_data)

        # decoding categorical columns
        input_data.replace({'Company Name': {'MG': 0, 'Hyundai': 1, 'Nissan': 2, 'TATA': 3, 'Volkswagen': 4,
                                            'Mahindra': 5, 'Maruti Suzuki': 6, 'Renault': 7, 'Toyota': 8,
                                            'Honda': 9, 'Kia': 10}}, inplace=True)

        input_data.replace({'Fuel Type': {'Diesel': 0, 'Petrol': 1}}, inplace=True)
        input_data.replace({'Tyre Condition': {'Needs Replacement': 0, 'New': 1, 'Used': 2}}, inplace=True)
        input_data.replace({'Owner Type': {'First': 0, 'Second': 1, 'Third': 2, 'Fourth': 3}}, inplace=True)
        input_data.replace({'Registration Number': {'DL-XX-XX-XXXX': 0, 'AP-XX-XX-XXXX': 1, 'MH-XX-XX-XXXX': 2,
                                                    'KA-XX-XX-XXXX': 3, 'TN-XX-XX-XXXX': 4, 'TS-XX-XX-XXXX': 5,
                                                    'JS-XX-XX-XXXX': 6, 'MP-XX-XX-XXXX': 7, 'KL-XX-XX-XXXX': 8,
                                                    'PB-XX-XX-XXXX': 9}}, inplace=True)
        input_data.replace({'Transmission Type': {'Automatic': 0, 'Automatic (Tiptronic)': 1,
                                                    'Manual': 2, 'unknown_transmission': 3}}, inplace=True)
        input_data.replace({'Registration Certificate': {'Not Available': 0, 'Available': 1}}, inplace=True)

        input_data.replace({'Car Name': {'Astor': 0, 'Gloster': 1, 'Etios': 2, 'Carens': 3, 'XUV300': 4, 'Eeco': 5, 'Jazz': 6, 'Hector Plus': 7, 'Thar': 8, 'Creta': 9, 'Sunny': 10, 'Vento': 11, 'i10': 12, 'Venue': 13, 'Camry': 14, 'Civic': 15, 'Altroz': 16, 'Micra': 17, 'Corolla': 18, 'Punch': 19, 'Bolero': 20, 'Hector': 21, 'Accord': 22, 'Ignis': 23, 'Verna': 24, 'Land Cruiser': 25, 'Tiago': 26, 'Kwid': 27, 'KUV100': 28, 'Tiguan': 29, 'Ioniq': 30, 'Safari': 31, 'Carnival': 32, 'Sonet': 33, 'Triber': 34, 'Ciaz': 35, 'Marazzo': 36, 'Kicks': 37, 'Nexon': 38, 'Glanza': 39, 'Dzire': 40, 'Innova Crysta': 41, 'T-Roc': 42, 'Polo': 43, 'Yaris': 44, 'Kiger': 45, 'Exter': 46, 'Omni': 47, 'City': 48, 'X-Trail': 49, 'Brezza': 50, 'i20': 51, 'Celerio': 52, 'Harrier': 53, 'Fortuner': 54, 'Taigun': 55, 'XUV700': 56, 'Scorpio': 57, 'WR-V': 58, 'Urban Cruiser': 59, 'Aura': 60, 'Baleno': 61, 'Magnite': 62, 'Alto': 63, 'Swift': 64, 'Innova': 65, 'Alcazar': 66, 'Amaze': 67, 'S-Presso': 68, 'Tigor': 69, 'Santro': 70, 'Seltos': 71, 'CR-V': 72, 'Ertiga': 73}}, inplace=True)

        # Print the input data before prediction
        print("Input Data Before Prediction:")
        print(input_data)

        # Make prediction
        predicted_price = model.predict(input_data)
        predicted_price = scaler.inverse_transform(np.array([[predicted_price[0]]]))[0][0]
        predicted_price = predicted_price**2
        
        
        return  round(predicted_price, 2) 
    
     # Return the predicted price

    predict_car_price = predict_car_price(make_year, mileage, company, car_name, fuel, tyre,owner, registration, transmission, certificate,Music_System,Sunroof,Alloy_Wheels,GPS,Leather_Seats)

    print(predict_car_price)
    print(Sunroof + Alloy_Wheels + GPS  + Leather_Seats  + Music_System )
    print("hello")


    
    return render_template('index.html', 
                           companies=company_options, 
                           fuels=fuel_options, 
                           tyres=tyre_options, 
                           owners=owner_options, 
                           registrations=registration_options,
                           transmissions=transmission_options,
                           certificates=certificate_options,
                           car_names=car_name_options,
                           accessories=accessory_weights,
                           predicted_price=predict_car_price)







if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))  # Default to 4000 if PORT is not set
    app.run(host='0.0.0.0', port=port)


# import time
# import http.client
# import asyncio

# async def handler(event, context):
#     url = 'yoursitehere.onrender.com'
#     connection = http.client.HTTPSConnection(url, timeout=10)  # Set timeout to 10 seconds

#     try:
#         connection.request('GET', '/')
#         response = connection.getresponse()

#         if response.status == 200:
#             print('Server pinged successfully')
#         else:
#             print(f'Server ping failed with status code: {response.status}')

#     except Exception as error:
#         print(f'Error occurred: {error}')

#     finally:
#         connection.close()

# async def start_pinging():
#     while True:
#         await handler({}, {})
#         time.sleep(45)  # Ping every 45 seconds

# # Start the pinging process
# asyncio.run(start_pinging())

