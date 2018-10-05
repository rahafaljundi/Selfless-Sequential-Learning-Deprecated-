#Backward hook to be inserted when computing the importance weights either using EWC or MAS
def compute_neuron_importance(self, grad_input, grad_output):
    
    if 'ReLU' in self.__class__.__name__:
        
        if hasattr(self, "task_neurons_importance"):#previous batch have computed neurons importance
            
            self.samples_size+=grad_input[0].size(0)
            self.task_neurons_importance+=torch.abs(torch.sum(grad_input[0],0))
            
        else:
            
            self.task_neurons_importance=torch.abs(torch.sum(grad_input[0],0))
            self.samples_size=grad_input[0].size(0)
            
#function that register the backward hook before computing the importance weights in EWC or MAS            
def compute_neurons_omega(model):
    handels=[]
    for name, module in model.module._modules.items():

        for namex, modulex in module._modules.items():
            #Register the backward hook
            handle=modulex.register_backward_hook(compute_neuron_importance)
            handels.append(handle)
            
    return model,handels

def set_neurons_omega_val(model):
    
    for name, module in model.module._modules.items():

        for namex, modulex in module._modules.items():
            
            if hasattr(modulex, "task_neurons_importance"):#has been registered for backward hook
                if hasattr(modulex, "neurons_importance"):#previous tasks had been learned and computed neurons importance
                  
                     modulex.neurons_importance+=modulex.task_neurons_importance/modulex.samples_size

                        
                else:
                    modulex.neurons_importance=modulex.task_neurons_importance/modulex.samples_size
                    
                del modulex.samples_size
                del modulex.task_neurons_importance
                  
            
    return model
