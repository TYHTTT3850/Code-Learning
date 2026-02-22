fn main() {
    let isSunny:bool = true;
    let windSpeed:f64 = 5.4;
    let temperature:i32 = 23;
    let solarPanelOutput:i32 = 9;
    let isCloudy:bool = false;
    
    let result: bool = isSunny && (windSpeed < 10.0) && (solarPanelOutput < 15) && (temperature > 20 || !isCloudy);
    
    println!("Checking conditions for solar energy production...");
    println!("1. Is it sunny? {}", isSunny);
    println!("2. Is wind speed safe? {}", (windSpeed < 10.0));
    println!("3. Can panels produce more? {}", (solarPanelOutput < 15));
    println!("4. Is temperature good OR no clouds? {}", (temperature > 20 || !isCloudy));
    println!("\\nFinal result - Good day for solar energy production: {}", result);
}