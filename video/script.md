# TeamTwins demo video: voiceover script (about 22 to 24 minutes)

Challenge rule: the video must be at least 20 minutes. This script is about 3,300 words, which is 22 to 24 minutes at a calm pace of 135 to 150 words per minute. Do not rush: if you read faster than 165 words per minute the video drops under 20 minutes. Do not cut below 20.

## How to record
- Record each scene as its own file: scene_01.wav, scene_02.wav and so on. Read only the lines under "VOICEOVER".
- Leave one second of silence at the start and end of every file.
- Speak slowly. Pause at full stops. Say numbers clearly: "one hundred and one million", "zero point nine three three".
- Quiet room, phone or mic six inches from your mouth, no echo. Do not worry about small slips: re-read the sentence and I will cut the bad take.
- Lines in [brackets] are cues for me, not for you to read.

The slide numbers refer to TeamTwins.pdf (21 slides). "LIVE" scenes show the running dashboard, which I will capture myself.

---

## Scene 01 | 0:00 to 1:00 | Slide 1 (title)
[Title slide, slow zoom, team names appear]

VOICEOVER:
Hello, and welcome. We are Team TeamTwins, from the Indira Gandhi Delhi Technical University for Women and GL Bajaj Institute of Technology and Management. I am Anisha Gupta, the team leader, and my teammate is Vibhor Sharma.

This is our submission to the Digital Twin Challenge 2026, organised by Happiest Health. Over the next twenty minutes we will show you a glucose digital twin for type 2 diabetes: a system that fuses a patient's health record with their wearable data, warns of a high or low glucose event up to two hours ahead, explains why, and ranks a whole panel of patients by who needs attention first.

We will cover the problem, the data, the architecture, the models, the results, a full live demonstration of the doctor dashboard, and, just as important, the limits of what we have built.

## Scene 02 | 1:00 to 2:30 | Slide 2 (problem)
[Stat circles animate one by one]

VOICEOVER:
Let us start with why this matters. India has about one hundred and one million people living with diabetes, and another one hundred and thirty six million with prediabetes, according to the ICMR INDIAB study. South Asians also tend to develop type 2 diabetes at a lower body mass index and at a younger age than most populations, so the risk starts early and runs for decades.

Now think about how care actually works. A patient meets a clinician a few times a year. But their glucose moves every few minutes. A dangerous high, above one hundred and eighty milligrams per decilitre, or a dangerous low, below seventy, can happen between visits and the clinician never sees it.

Continuous glucose monitors and smartwatches already record data every five minutes, but that data is not joined to the patient's labs, diagnoses, medicines or genetics. And no doctor can read thousands of traces. So risk is found after the event, not before it.

Our question is simple. Can a patient's record, fused with their wearable streams, warn of a glucose event up to two hours ahead, explain why, and tell the clinic who needs attention first?

## Scene 03 | 2:30 to 4:00 | Slide 3 (clinical definitions)
[Glucose zone scale, then the four definition tiles]

VOICEOVER:
Before the technology, we fixed the clinical definitions, because a prediction is only as good as its target.

We follow the international consensus on continuous glucose monitoring, which clinicians use to judge whether a patient's diabetes is under control. The healthy range is seventy to one hundred and eighty. Below seventy is a level one low, and below fifty four is a level two low.

An event, for us, is not a single spike. It must last at least fifteen minutes, which means three consecutive five minute readings out of range. This ignores sensor noise. An onset only counts if the patient was fully in range for the previous hour, so we predict new events, not ones already underway.

The label looks only at the next five to one hundred and twenty minutes, and every feature uses data up to the present moment and nothing later. And an alert is not fired on one reading: risk must be above the threshold now and ten minutes earlier.

One more fact to keep in mind. Lows are rare: only two point seven percent of in range moments lead to one. So every precision number we show later must be read against that base rate.

## Scene 04 | 4:00 to 5:30 | Slides 4 and 5 (organiser and brief)
[Organiser slide, then requirements table with ticks appearing]

VOICEOVER:
This challenge is organised by Happiest Health, a wellness enterprise based in Bengaluru, announced in April 2022 and promoted by Ashok Soota. It has four divisions: knowledge, diagnostics, healthcare services and wellness. Mr Soota previously helped build Wipro's IT business and was founding chairman of MindTree and Happiest Minds. The summit's own programme talks about continuous health modelling and the opportunities and limits of A I, and that is exactly the spirit of our project: a continuous health model, with its limits stated honestly.

The brief asks for specific things, and this table maps each one to our evidence. Static data: a simulated health record with demographics, diagnoses, labs and genetic markers. Dynamic data: wearable time series for glucose, heart rate variability, sleep and steps. A fusion of the two. A model that predicts an adverse event. A conceptual doctor dashboard. And synthetic or open data only, in line with the D P D P Act and HIPAA.

We meet every one, and the checklist on the right shows the submission items: public repository, README, M I T licence, architecture diagram, this presentation, and this video.

## Scene 05 | 5:30 to 7:30 | Slides 6 and 7 (data)
[Static data slide with donut and bars, then wearable streams table and the EHR to physiology flow]

VOICEOVER:
Now the data. We use no real patient data at any point.

The health record is built from Synthea, an open source synthetic patient generator. Synthea gives us demographics, diagnosis dates, body measurements, lipids and smoking history. Our own code adds what Synthea lacks: four to eight lab visits in the two years before the anchor date, fasting insulin, medicines with clinical rules, complications, and five real diabetes risk variants, such as T C F 7 L 2 and F T O, with simulated genotypes and a polygenic score. Every genetic field is labelled simulated.

We built two cohorts. A sample of fifty patients over seven days for fast tests, and the full cohort of eight hundred patients over fourteen days. The full mix is thirty eight percent type 2 diabetes, thirty one percent prediabetes and thirty one percent no diabetes.

The wearable streams come next. They run at native device rates: a Dexcom style glucose reading every five minutes with sensor noise, warm up gaps and dropouts of fifteen to one hundred and twenty minutes; a smartwatch with heart rate, heart rate variability, steps, sleep stages, oxygen saturation and wrist temperature; and manual insulin and carbohydrate logs that are deliberately under recorded, because real patients forget.

The key design choice is the link between the two. Each patient's glucose simulator is parameterised from their own record. Insulin resistance, B M I, kidney function, medicines and genetics all change how that patient's glucose behaves. So the record carries real signal for the fusion model to discover. We also keep the simulator's one minute ground truth, for validation only. It is never a model input.

## Scene 06 | 7:30 to 9:30 | Slides 8 and 9 (pipeline and architecture)
[Pipeline chevrons, then the full architecture diagram; camera pans lane by lane, highlighting each lane as it is described]

VOICEOVER:
This is how it all fits together. The pipeline has five stages, and each has a test.

Layer one is data. The run pipeline script reads the profile from config, Synthea creates the patients, and build E H R creates the record tables.

Layer two is simulation. Derive params turns the record into physiology parameters. The simulator uses the simglucose model with a controller standing in for beta cell secretion, plus a behaviour model for meals, steps, sleep and heart rate. It writes the raw device streams, and the hidden ground truth.

Layer three is ingestion. The aligner puts every stream onto one five minute timeline. It uses the nearest reading within two and a half minutes, fills only gaps of fifteen minutes or less, and keeps flags saying what was filled. Then fuse with E H R joins the record, using only labs, conditions and medicines dated before that day. A unit test proves no future lab ever reaches a timestamp.

Layer four is modelling. Features, the ablation study, and the event models, with every result saved to the results folder.

Layer five is serving. Score stream rebuilds features from only the history received so far and scores with a model that never saw that patient. The alert rule decides, and the dashboard shows it.

Blue boxes are code, cream boxes are data files, and the green arrows are the data flow.

## Scene 07 | 9:30 to 11:30 | Slides 10 and 11 (features and models)
[Feature group tiles, model table, then the two top feature bar charts]

VOICEOVER:
The event model uses one hundred and two features, in five groups, all strictly causal.

Glucose dynamics: lagged values, rates of change, acceleration, smoothed slopes, and minutes to reach seventy or one hundred and eighty at the current slope. Insulin and carbohydrates on board, from the logs, using four hour and one hour half lives. Personal history: how many highs and lows the patient had in the last twenty four and seventy two hours, and at this same hour on previous days. Physiology: heart rate, steps, sleep, resting heart rate trend, oxygen saturation and wrist temperature. And the health record: age, HbA1c, lipids, kidney function, medicines, genetic counts and more.

The models are LightGBM. Two classifiers predict glucose above one eighty and below seventy within two hours: five hundred trees, learning rate zero point zero three, fifteen leaves, and at least three hundred samples per leaf. We regularised them heavily on purpose. Record columns are constant for each patient, so loose trees memorise training patients. Before tuning, that cost us about three hundredths of average precision on unseen patients.

Quantile models forecast glucose one and two hours ahead, and we widen the band so it truly covers eighty percent of outcomes on held out patients.

Everything is evaluated with five fold cross validation grouped by patient, so no patient appears in both training and test. The forecast uses a technique called conformal calibration: after training, we measure how wrong the quantile model is on patients it has not seen, and widen the band by exactly that amount, which is why the stated eighty percent coverage is a measured fact. For explanations, each fold keeps its trained booster, and the dashboard reads the contribution of every feature to a single prediction, so every alert can be explained from a model that never saw that patient.

On the chart you can see what the models look at: minutes since the last meal, recent glucose and time of day lead. In the fused model, HbA1c and fasting glucose enter the top ten, which is the record acting as a prior.

## Scene 08 | 11:30 to 14:00 | Slides 12 to 15 (results)
[Results chart, table, trade off curve, forecast chart, ablation, sample results]

VOICEOVER:
Now the results, on eight hundred patients and one point two million test moments from patients the model never saw.

For glucose above one eighty, our model reaches an AUC of zero point nine three three, against zero point seven four eight for a simple trend rule. Average precision is zero point six seven. At the balanced setting it catches eighty nine percent of events with a median warning of eighty minutes, at about one point eight five alerts per patient per day.

For lows, AUC is zero point nine five six, against zero point eight four two for the trend rule. It catches seventy four percent with a median warning of seventy minutes. Average precision is zero point four seven, which is eighteen times the base rate.

A clinic can choose the trade off. The sensitive setting catches ninety six to ninety seven percent of events but raises more false alarms. The precise setting gives about fifty four percent alert precision for highs, and for lows it gives only point two five alerts a day with fifty five percent real, but misses about forty eight percent of lows.

Forecasts: median error is eleven point nine milligrams per decilitre at one hour, against seventeen point four for simply repeating the last value, and sixteen point one at two hours against twenty six point four. The eighty percent band really does cover eighty percent.

Now an honest answer to a hard question: does the health record help? After a patient has one day of glucose history, no. Personal history carries what the record knew, and the fused and wearable only models tie. But in the first twenty four hours, before that history exists, the record helps for highs, with AUC zero point nine one six against zero point nine zero five. And against a model without personal history, the record cuts sixty minute error from thirteen point five to twelve point eight. So the record's value here is a day one prior.

Finally, the numbers committed in the repository are from the fifty patient sample run, which regenerate with one command. They are lower, as you would expect with fewer patients, and we show them for transparency.

## Scene 09 | 14:00 to 18:00 | Slide 16 then LIVE dashboard
[Slide 16 for ten seconds, then switch to the live Streamlit dashboard. Capture each page in order.]

VOICEOVER:
Now the part that matters to a doctor: the dashboard. It is a working Streamlit app, not a mock up.

[LIVE: Cohort overview]
This is the cohort overview, and we have switched to the full dataset. The top cards show eight hundred patients, of whom three hundred and four have type 2 diabetes and two hundred and forty eight have prediabetes. The median HbA1c is six point one percent and the median body mass index is thirty point two. Below, three histograms show how HbA1c, body mass index and the simulated polygenic risk score are spread across the panel, and a filterable table lists every patient. Think of it as the clinic's first screen.

[LIVE: Triage board]
This is the triage board, the heart of the design. The twin ranks the whole panel by predicted risk of a glucose event in the next two hours. The highest risk patients are on top. You can see in the sidebar the alert sensitivity: sensitive, balanced or precise. Changing it changes how many patients are flagged, and the measured performance of each setting is on the model results page. In the snapshot on screen, the panel has eight hundred patients. Two hundred and eight are in alert right now, which is twenty six percent of the panel. Two hundred and two of those alerts are for high glucose and six are for low glucose. The expected workload card turns this into a planning number: about two hundred and forty alerts per day for every hundred patients. A doctor can supervise a large panel by looking only at who is in alert right now. There is also a time slider, so the clinic can move the snapshot clock and watch the ranking change.

[LIVE: Patient view, Glucose and wearables tab]
Opening a patient, we see a summary strip first. This synthetic patient is a forty five year old man with type 2 diabetes, an HbA1c of seven point one, a body mass index of twenty one point eight, kidney function, or eGFR, of eighty five, an insulin resistance score, HOMA IR, of two point three eight, and a simulated genetic risk of zero point eight two. Below that, the glucose cards show a mean of one hundred and sixty eight, sixty percent of time in range and forty percent above range. The risk cards say that at the selected moment the chance of glucose above one eighty within two hours is eighty nine percent, with an alert raised, glucose now one hundred and seventy three, and a forecast of one hundred and ninety three at one hour. Then we see the glucose trace, heart rate, steps and sleep. The shaded band is the one hour or two hour forecast, with its eighty percent uncertainty, and the dashed lines mark one hundred and eighty and seventy. Orange triangles mark meals the patient logged, and some meals go unlogged, as in real life.

[LIVE: Why flagged]
Here is why the patient was flagged. These bars come from the model's explanation, and each driver is tagged: wearable, or health record. So the doctor can see whether a falling glucose trend, last night's sleep, or the patient's HbA1c pushed the risk up.

[LIVE: What if panel]
The what if panel lets us toggle a medicine or activity level and watch the risk move. We must be careful: this shows associations learned from synthetic data. It is not causal treatment advice, and the screen says so.

[LIVE: Health record, Genetics, Devices tabs]
The other tabs hold the health record: conditions, lab trends and medications; the simulated genetics, with the five variants and the polygenic score; and the devices, with their data rates and quality.

[LIVE: Live monitor]
Now the live monitor. You can pick a patient, set the speed, restart, or jump straight to the next alert, which is what we do here. It replays a simulated real time feed. The twin only sees data that has arrived so far, rescored at every tick, and it looks two hours ahead. Watch the risk rise before the glucose crosses the line. And the alert log below marks every alert as confirmed or false once two hours have passed, so the clinic can see the true record of alerts.

[LIVE: Model results]
And finally the model results page, which asks the key scientific question: does adding the health record help predict glucose sixty minutes ahead? The headline cards show the average error: seventeen point four for simply repeating the last reading, thirteen point five for the wearable only model, twelve point nine when the health record snapshot is added, and twelve point eight with the full history and genetics. These are tested on eight hundred unseen patients. Below are expandable sections with every metric, the paired statistical tests per patient, and which features mattered most. Further down you see the event metrics, average precision against the trend rule, the forecast error and band coverage, and every alert operating point, with the limitations written right beside them so nobody can miss them.

## Scene 10 | 18:00 to 19:00 | Slide 17 (real time and tests)
[Real time flow, then the test table]

VOICEOVER:
A word about trust. The live path is not a separate system. The same function, score stream, rebuilds features from the history received so far and scores with a held out model. A test proves that streamed features and risks equal the batch ones, so there is no look ahead.

In total, thirteen automated tests guard our claims: the record is internally consistent, clinical rules hold, device streams keep native rates, no future lab leaks into a timestamp, labels use only the future window, and live scoring equals batch scoring. Everything can be reproduced from the repository: one command for the sample run in minutes, one for the full run in about an hour, and one more command for the dashboard. The fifty patient sample is committed with the repository, so a reviewer can open the dashboard and run the tests within minutes of cloning it. The event stage alone, which trains all the classifiers and forecasters on the full cohort, took about five and a half minutes on a ten core laptop when we reran it for this video.

## Scene 11 | 19:00 to 20:30 | Slides 18 and 19 (impact, limitations, roadmap)
[Funnel, alert volume gauges, limitations list, roadmap]

VOICEOVER:
What does this mean in practice? At the balanced setting a monitored patient triggers about two point four alerts a day, one point eight five high and half a low. About half of high alerts and about a third of low alerts are followed by a real event. That is too many to send to a doctor, so the design is tiered. The patient is alerted first, and the clinician sees only who is in alert right now. A small team can supervise a large panel between visits, and each clinic sets its own dial between sensitive and precise. A clinic with few staff would choose the precise setting and accept missing some events. A clinic with a dedicated monitoring nurse could choose the sensitive setting and catch nearly every event, at the price of more calls. The model is the same; only the threshold moves.

Now the limitations, stated plainly. The simulator is driven by the health record, so our fusion gain shows the pipeline can recover that link, not that it exists in real patients. Hypoglycaemia is the weaker result: about two thirds of low alerts are false at the balanced setting, and what remains looks like unlogged insulin and meals that no feature can recover. The glucose model is a generic glucose insulin system, not a validated type 2 model. Thresholds are mildly optimistic. And what if answers are associations, not causes. This is research code, not a medical device.

The roadmap follows from that: validate on real, consented data such as MIMIC IV and an open CGM dataset; calibrate thresholds to the clinical cost of missed and false alarms with clinicians; connect a real device feed through an adapter; and extend the twin to hypertension or cardiovascular risk.

## Scene 12 | 20:30 to 21:15 | Slides 20 and 21 (codebase and close)
[Codebase bars, run commands, then the closing slide]

VOICEOVER:
The whole system is about sixteen hundred lines of Python, in separate modules with one entry point. The architecture diagram, this presentation, the M I T licence and the full README are in the public repository, and the commands on screen reproduce everything.

To summarise: we fused a synthetic health record with simulated wearable data, predicted glucose events two hours ahead with a median warning of seventy to eighty minutes, explained every alert, built a doctor dashboard around triage, and told you honestly where the evidence stops.

Same patient, earlier warning. Thank you to Happiest Health for this challenge, and thank you for watching. We are Team TeamTwins.

---

## What I will do after you send the audio
1. Match each scene file to its slides, with slow zooms and highlights on the part being spoken.
2. Capture the live dashboard pages and the live monitor myself, and sync them to Scene 09.
3. Add lower thirds (names, section titles), callout boxes on key numbers, and captions throughout so the video is self explanatory without sound.
4. Add a short intro and outro card, quiet background music under the voice, loudness normalisation, and export a 1080p MP4 you can upload as an unlisted YouTube video.
