#%%
import kagglehub
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Download latest version
#path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
#print("Path to dataset files:", path)
#%%
class StudentPerformancePredictor():
    def __init__(self):
        self._gather_data()
        self.preprocessing()
        self.build_model()

    def _gather_data(self) -> None:
        print("data gathering...")
        student_data_path : str = "/Users/gabbitaylor/.cache/kagglehub/datasets/nikhil7280/student-performance-multiple-linear-regression/versions/1/Student_Performance.csv"
        student_data_df : pd.DataFrame = pd.read_csv(student_data_path)

        self.student_data_df = student_data_df

    def preprocessing(self) -> pd.DataFrame:
        print("preprocessing...")
        clean_student_data_df = self.student_data_df

        if self._check_null():
            clean_student_data_df = clean_student_data_df.dropna()
        
        elif self._check_duplicates():
            clean_student_data_df = clean_student_data_df.drop_duplicates()
        
        self.clean_student_data_df = self.transform_categorical(clean_student_data_df)

    def build_model(self):
        print("build model...")
        target_variable = 'Performance Index'

        print('get baseline score')
        baseline_preds = [self.clean_student_data_df["Performance Index"].mean()] * len(self.clean_student_data_df)
        baseline_mse = mean_squared_error(self.clean_student_data_df["Performance Index"], baseline_preds)
        print("baseline mse:", baseline_mse)

        X = self.clean_student_data_df.drop(target_variable, axis=1)
        y = self.clean_student_data_df[target_variable]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = self.model_selection(X, y)
        #model.fit(x_train, y_train)

        self.evaluation(model, X, y)
    
    def model_selection(self, X, y):
        print("model selection...")
        
        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Support Vector Regression": SVR(kernel='linear'),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
        }

        # Evaluate models using cross-validation (e.g., 5-fold)
        cv_results = {}
        for model_name, model in models.items():
            # Compute cross-validation scores (negative MSE)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_log_error')
            
            # Store the mean of the cross-validation scores (lower is better)
            mse = -scores.mean()
            cv_results[model_name] = mse

        # Display cross-validation results
        lowest_mse = 1000
        for model_name, mse in cv_results.items():
            print(f"{model_name}: CV MSE = {mse}")
            
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = model_name

        print(f'best model: {best_model}')
        return models[best_model]
    
        """
        # Train models and store results
        results = {}
        for model_name, model in models.items():
            # Fit the model
            model.fit(x_train, y_train)
            
            # Make predictions
            y_pred = model.predict(x_test)
            
            # Calculate performance metric (e.g., Mean Squared Error)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store the results
            results[model_name] = mse
        
        # Display model performances
        lowest_mse = 1000
        for model_name, mse in results.items():
            print(f"{model_name}: MSE = {mse}")
            
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = model_name

        print(f'best model: {best_model}')
        return models[best_model]
        """

    def evaluation(self, model, X, y):
        print("evaluation...")
        #y_pred = model.predict(x_test)
        #final_mse = mean_squared_error(y_test, y_pred)

        scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_log_error')
        final_mse = -scores.mean()

        self.scores = scores
        self.y = y

        print("MSE:", final_mse)

    def _check_null(self) -> int:
        null_count = self.student_data_df.isnull().sum().sum()

        return null_count

    def _check_duplicates(self):
        dup_count = self.student_data_df.duplicated().sum()

        return dup_count
    
    def transform_categorical(self, clean_student_data_df):
        encoder = LabelEncoder()

        categorical_columns = clean_student_data_df.select_dtypes(include=['object']).columns

        for ccol in categorical_columns:
            clean_student_data_df[ccol+"_encoded"] = encoder.fit_transform(clean_student_data_df[ccol])
            clean_student_data_df.drop(columns=ccol, inplace=True)

        return clean_student_data_df

if __name__ == "__main__":
    spp = StudentPerformancePredictor()

#%%
