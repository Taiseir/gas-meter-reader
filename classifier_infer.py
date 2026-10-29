"""
classifier_infer.py - stub.

The original module was never exported with the rest of the Spring 2026 code.
Tracing the Roboflow workflow showed the classifier it called was
`car-colors-1smyc/5`, a public demo model for identifying the colour of cars,
weighted 1.3 in the ensemble - higher than any of the three geometric methods.

Returning None here makes the ensemble skip that vote entirely, which also
removes the last step that required an internet connection.
"""


def predict_digit_with_classifier(dial_img, dial_name):
    """No classifier. The ensemble falls back to ray, contour and line."""
    return None, 0.0
