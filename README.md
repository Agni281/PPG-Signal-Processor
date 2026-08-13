# PPG-Signal-Processor
In remote clinics, medical equipment is low cost and poorly-shielded. When patient is hooked on a PPG sensor to measure their heart rate, the signal gets corrupted due to
1. High-Frequency Noise: Electrical Grid Hum, Motion Artifacts (muscular noise)
2. Low-Frequency Drift: Patient is breathing or moving, causing the baseline of the wave to drift up and down.
   
An intelligent signal processing pipeline that takes raw 1-dimensional streams of voltage readings and checks if the data is salvageable. 
1D Convolutional UNet model reconstructs the clean heartbeat wave, calculates confidence in the cleanup, finds exact pulse peaks, and outputs a diagnostic conclusion. 

Trained upon PhysioNet's CSV files the model is able to handle real life data and denoise both synthetic and real world.

Try it here: (https://ppg-signal-proceappr-b5appmnqj2tmimeap572trp.streamlit.app/)
