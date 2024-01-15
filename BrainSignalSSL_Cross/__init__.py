################################################
#### THIS IS THE FINAL VERSION OF *MBrain*. ####
################################################

# We will do cross-patient seizure detection in this model.
# For example, if there are five patients from patient-A to Patient-E,
# self-supervised model MBrain will be pretrained in datasets including Patient-A and Patient-B and Patient-C.
# After pretraining stage ends, Pateint-A/Patient-B/Patient-C will be used for downstream training and Patient-D will be used for validation.
# Finally, We will test in the Patient-E.