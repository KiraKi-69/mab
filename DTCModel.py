class DTCModel(object):  
    
    def __init__(self):
        
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import OrdinalEncoder
        
        self.categorical_features = [
            "person_home_ownership",
            "loan_intent",
            "city",
            "state",
            "location_type",
        ]
        
        self.encoder = joblib.load("encoder.pkl")
        
        print("Encoder loaded")
        
        self.model = joblib.load("DTC.pkl")
        
        print("Model loaded")

    def predict(self,X,features_names):

        df = pd.Dataframe(X, columns=features_names)
        
        df[self.categorical_features] = self.encoder.transform(df[self.categorical_features])
        df = df.reindex(sorted(df.columns), axis=1)
        
        predictions = self.model.predict(df)
        
        return predictions

    def send_feedback(self,features,feature_names,reward,truth,routing):
        """
        Handle feedback

        Parameters
        ----------
        features : array - the features sent in the original predict request
        feature_names : array of feature names. May be None if not available.
        reward : float - the reward
        truth : array with correct value (optional)
        """
        print("Send feedback called")
        return []