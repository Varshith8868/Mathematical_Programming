from django import forms

class LPSolverForm(forms.Form):
    c1 = forms.FloatField(label="Objective Function Coefficient for x1", required=True)
    c2 = forms.FloatField(label="Objective Function Coefficient for x2", required=True)

    num_constraints = forms.IntegerField(
        label="Number of Constraints",
        min_value=1,
        required=True,
        initial=2
    )


# transportation/forms.py
# from django import forms

# class TransportationForm(forms.Form):
#     # Input the number of sources and destinations
#     num_sources = forms.IntegerField(label='Number of Sources (Supply)', min_value=1, initial=3)
#     num_destinations = forms.IntegerField(label='Number of Destinations (Demand)', min_value=1, initial=3)

#     # Initially empty supply, demand, and cost fields will be generated dynamically via JavaScript
#     supply = forms.CharField(widget=forms.HiddenInput(), required=False)
#     demand = forms.CharField(widget=forms.HiddenInput(), required=False)
#     cost = forms.CharField(widget=forms.HiddenInput(), required=False)
