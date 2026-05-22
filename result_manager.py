import os
from datetime import datetime

class ResultManager:
    def __init__(self, output_file):
        self.output_file = output_file
        
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


    def _get_header(self):
        return f"""
=====================================================
                       MOSAIC                                                                     
     A multi-model benchmarking toolkit for ML-CV                           
               with PCA/UMAP reduction                                     
             and GridSearch optimization
-----------------------------------------------------              
          Models: SVC, RandomForest, XGBoost
          Exporter CLI: export_best_model.py                            
-----------------------------------------------------
Developer: Flavio F. Contreras-Torres
Version: v0.1.0 - May, 2025. Oviedo
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-----------------------------------------------------
GitHub: https://github.com/NanoBiostructuresRG
=====================================================

"""
    

    def write_results(self, content):
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(self._get_header()) 
                f.write(content)
        except Exception as e:
            print(f"Error writing results: {e}")