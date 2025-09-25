import streamlit as st
from openai import OpenAI
import requests
import json

# Configuration
st.title("Lab 5 - Weather & What to Wear Bot")
st.caption("Get weather information and clothing suggestions for any city!")

# API Keys from secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"].strip()
WEATHER_API_KEY = st.secrets["openweather"]["api_key"].strip()

# Initialize OpenAI client
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
client = st.session_state.openai_client

# =========================
# Lab 5a: Weather Data Function
# =========================
def get_current_weather(location, api_key=WEATHER_API_KEY):
    """
    Get current weather data for a location using OpenWeatherMap API.
    Returns weather information including temperature, description, etc.
    """
    if "," in location:
        location = location.split(",")[0].strip()
    
    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={api_key}"
    url = urlbase + urlweather
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract temperatures & Convert Kelvin to Celsius
        temp = data['main']['temp'] - 273.15
        feels_like = data['main']['feels_like'] - 273.15
        temp_min = data['main']['temp_min'] - 273.15
        temp_max = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        
        # Additional weather info
        weather_desc = data['weather'][0]['description']
        weather_main = data['weather'][0]['main']
        wind_speed = data.get('wind', {}).get('speed', 0)
        
        return {
            "location": location,
            "temperature": round(temp, 2),
            "feels_like": round(feels_like, 2),
            "temp_min": round(temp_min, 2),
            "temp_max": round(temp_max, 2),
            "humidity": round(humidity, 2),
            "description": weather_desc,
            "main": weather_main,
            "wind_speed": wind_speed
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        st.error(f"Error parsing weather data: {e}")
        return None


# =========================
# Lab 5b: OpenAI Function/Tool Definition
# =========================
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather information for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'New York' or 'London, UK'"
                }
            },
            "required": ["location"]
        }
    }
}

def call_openai_with_tools(user_message, model="gpt-4o-mini"):
    """
    Call OpenAI API with weather tool capability.
    Returns the assistant's response after potentially calling the weather function.
    """
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful weather assistant. When asked about weather, use the get_current_weather function. Provide clothing suggestions and advice about outdoor activities like picnics based on the weather conditions. If no location is provided, use Syracuse NY as default."
        },
        {"role": "user", "content": user_message}
    ]
    
    # First API call with tools
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    # Check if the model wants to call a function
    if assistant_message.tool_calls:
        # Add the assistant's message to conversation
        messages.append(assistant_message)
        
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "get_current_weather":
                # Parse the location from the function call
                function_args = json.loads(tool_call.function.arguments)
                location = function_args.get("location", "Syracuse NY")
                
                # Call our weather function
                weather_data = get_current_weather(location)
                
                if weather_data:
                    # Add the function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(weather_data)
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: Could not retrieve weather data for the specified location."
                    })
        
        # Second API call with the function results
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return second_response.choices[0].message.content, weather_data
    else:
        # No function call needed
        return assistant_message.content, None

# =========================
# Main Application Interface
# =========================
st.subheader("🌤️ Weather & Clothing Advisor")

# Model selection
model_choice = st.selectbox(
    "Choose AI Model", 
    ["gpt-4o-mini", "gpt-4o"], 
    index=0
)

# City input
city_input = st.text_input(
    "Enter a city name:", 
    placeholder="e.g., Syracuse NY, London England, Tokyo Japan",
    help="If no city is provided, Syracuse NY will be used as default"
)

# Submit button
if st.button("Get Weather & Clothing Advice", type="primary"):
    if not city_input.strip():
        city_input = "Syracuse NY"
        st.info("No city provided, using Syracuse NY as default.")
    
    with st.spinner("Getting weather data and generating suggestions..."):
        # Create user message
        user_message = f"What's the weather like in {city_input}? Give me clothing suggestions and tell me if it's a good day for a picnic."
        
        # Get response from OpenAI with weather tool
        response, weather_data = call_openai_with_tools(user_message, model_choice)
        
        # Display results
        if weather_data:
            # Show weather data in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"🌡️ Temperature in {weather_data['location']}", 
                    value=f"{weather_data['temperature']}°C",
                    delta=f"Feels like {weather_data['feels_like']}°C"
                )
                
            with col2:
                st.metric(
                    label="💧 Humidity", 
                    value=f"{weather_data['humidity']}%"
                )
            
            # Weather description
            st.info(f"**Current conditions:** {weather_data['description'].title()}")
            
            # Additional details
            with st.expander("📊 Detailed Weather Info"):
                st.write(f"**High:** {weather_data['temp_max']}°C")
                st.write(f"**Low:** {weather_data['temp_min']}°C")
                st.write(f"**Wind Speed:** {weather_data['wind_speed']} m/s")
                st.write(f"**Weather Type:** {weather_data['main']}")
        
        # Show AI response
        st.subheader("🤖 AI Suggestions")
        st.write(response)