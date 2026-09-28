"""Export the deterministic AME actor with identical map/proprio ordering."""
import copy
import torch
from torch import nn
class ActorExport(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.cnn=copy.deepcopy(p.map_cnn)
        self.embedding=copy.deepcopy(p.actor_proprio_embedding)
        self.mha=copy.deepcopy(p.mha)
        self.actor=copy.deepcopy(p.actor)
    def forward(self,obs):
        proprio=obs[:,:45]
        points=obs[:,45:].reshape(-1,21,33,3).permute(0,3,1,2)
        features=self.cnn(points).flatten(2).transpose(1,2)
        encoded,_=self.mha(self.embedding(proprio).unsqueeze(1),features,features,need_weights=True)
        return self.actor(torch.cat((encoded.squeeze(1),proprio),-1))
def export_policy(p,example,out):
    p.eval()
    model=ActorExport(p).cpu().eval()
    x=example.detach().cpu()
    with torch.no_grad():
        torch.testing.assert_close(model(x).to(example.device),p.act_inference({'policy':example}),atol=2e-5,rtol=2e-5)
        torch.jit.script(model).save(str(out/'policy.pt'))
        torch.testing.assert_close(torch.jit.load(str(out/'policy.pt'))(x),model(x))
        torch.onnx.export(model,x[:1],str(out/'policy.onnx'),input_names=['observations'],output_names=['raw_actions'],dynamic_axes={'observations':{0:'batch'},'raw_actions':{0:'batch'}},opset_version=17,dynamo=False)
    import onnxruntime as ort
    import numpy as np
    session=ort.InferenceSession(str(out/'policy.onnx'),providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None,{'observations':x.numpy()})[0],model(x).detach().numpy(),rtol=1e-4,atol=2e-5)
