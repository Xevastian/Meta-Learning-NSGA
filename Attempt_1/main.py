# Example usage of MetaLearningNSGA2

from nsga2 import MetaLearningNSGA2

def main():
    # Initialize with dataset
    ml = MetaLearningNSGA2(data_path='Spam.csv', pop_size=5, generations=3)
    
    # Run the algorithm
    pareto = ml.run()
    
    # List models
    ml.list_pareto_models()
    
    # Example: Get config for first model
    if pareto:
        config = ml.get_model_config(0)
        print(f"\nModel config:\n{config}")
        
        # Get confusion matrix
        cm = ml.get_confusion_matrix(0)
        print(f"\nConfusion matrix:\n{cm}")

if __name__ == '__main__':
    main()