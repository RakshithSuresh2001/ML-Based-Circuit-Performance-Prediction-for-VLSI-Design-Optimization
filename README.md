Traditional VLSI design requires running thousands of SPICE simulations to optimize circuit parameters (transistor width, drive strength, etc.), which takes hours to days.

I built a machine learning pipeline that predicts circuit performance instantly from design parameters.

<img width="482" height="251" alt="R2_and_MAE_Summary" src="https://github.com/user-attachments/assets/022e2f8a-0e82-49a9-aefb-47a66e24b28e" />

<img width="1025" height="767" alt="New_Visualisation" src="https://github.com/user-attachments/assets/5b1e0ed8-8bdd-4c35-b56e-c130d7783857" />

<img width="499" height="407" alt="Trend_Analysis" src="https://github.com/user-attachments/assets/c7adeec9-a98c-4543-85d6-7c5d8eeb56b5" />

✅ Data Extraction: Parsed Nangate 45nm Liberty files (37+ standard cells)

✅ Feature Engineering: Created 7 derived features (PDP, area efficiency, normalized metrics)

✅ Model Training: Compared 5 algorithms (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)

✅ Model Selection: Random Forest achieved best performance (R² > 0.95, MAE < 5ps)
✅ Validation: Tested across multiple scenarios including drive strength sweeps

Results:

✅ 95%+ accuracy in predicting circuit delay

✅ <5 picosecond error (excellent for 45nm technology)

✅ 100× speedup over SPICE simulation

✅ Robust across design space (X1 to X32 drive strengths)

Technologies used: 

✅Python

✅scikit-learn

✅pandas

✅HSPICE

