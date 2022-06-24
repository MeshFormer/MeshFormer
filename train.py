import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random as rd

from tensorboardX import SummaryWriter
from sklearn.preprocessing import OneHotEncoder
from enum import Enum


from util.util import *
from util.logger import *
from MeshTransformer.HeteroDireGraph import *
from MeshTransformer.MeshTransformer import *
from MeshTransformer.FocalLoss import focal_loss


class MeshTransformer:
    def __init__(self, args, hetero_dir, writer_dir, in_dim=38, n_hid = 400, n_heads = 8,
                        n_layers=5, dropout =0.2, hirechy_layer=3,  loss_type='focal',
                        lr_schedual_type = 'Step', load_Path=None,
                        test_num=1,  task_name = "", device='0'
                    ):
        self.loadPath = load_Path
        self.writerDir = writer_dir
        self.args = args

        self.device = 'cuda:' + str(device)
        self.writer = SummaryWriter(self.writerDir)
        self.beginwriter = False
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.data = []
        self.colors = torch.randint(256,(1000,3)).to(self.device)
        self.num_types  = -1
        self.num_relations = -1
        self.num_label = -1
        self.task_name = task_name
        self.test_num = test_num 
        self.log = Logger(sys.path[-1])


        tdata = []
        tlabels = []
        alldata_heteo = [os.path.join(hetero_dir, i) for i in os.listdir(hetero_dir) if
                         len(i.split('.')) == 1 or i.split('.')[1] != 'txt' or i.split('.')[1] != 'html']
        self.data_num = []
        for i in os.listdir(hetero_dir):
            if len(i.split('.')) == 1 or i.split('.')[1] != 'txt' or i.split('.')[1] != 'html':
                self.data_num.append(int(i.split('.')[0]))

        for i in range(len(alldata_heteo)):
            hetero_graph = load_cache(alldata_heteo[i])
            self.log.info('find file:'+str(alldata_heteo[i])+', and load')
            hetero_graph.select_file = alldata_heteo[i]
            hetero_graph.hgraph.add_triangles(hetero_graph.mesh.data['mesh'].faces)
            tdata.append(hetero_graph)
            tlabels+=hetero_graph.get_label()
        self.enc.fit(np.array(tlabels).reshape(-1,1))


        for hetero_graph in tdata:
            hetero_graph.set_one_hot_label(self.enc)
            hgraph, ylabels, nlabels, face_area = hetero_graph.train_all_data()
            self.data.append([hgraph, ylabels, nlabels, hetero_graph.select_file, face_area])
            if in_dim != np.shape(hgraph.node_feature[hgraph.get_types()[0]])[1] + len(hgraph.get_types()):
                self.log.info(
                            'change indim from: '+ str(in_dim) 
                            +' to: '
                            + str(np.shape(hgraph.node_feature[hgraph.get_types()[0]])[1])
                         )
                
                in_dim = np.shape(hgraph.node_feature[hgraph.get_types()[0]])[1] + len(hgraph.get_types())
            if self.num_types == -1 or self.num_relations == -1 or self.num_label == -1 :
                self.num_types = len(hgraph.get_types())
                self.num_relations = len(hgraph.get_meta_graph()) + 1
                self.num_label = np.shape(ylabels)[1]
            else:
                assert( self.num_types == len(hgraph.get_types()))
                assert( self.num_relations == len(hgraph.get_meta_graph()) + 1)
                assert( self.num_label == np.shape(ylabels)[1])
        rd.shuffle(self.data)
        self.folders = []
        for i in range(10):
            start = int(i * len(self.data) / 10 )
            end =  int((i+1) * len(self.data) / 10 ) 
            self.folders.append(self.data[start:end])
            
            

        self.mblocks = MBLocks(
                                in_dim=in_dim,
                                n_hid=n_hid,
                                n_heads=n_heads,
                                n_layers=n_layers,
                                dropout=dropout,
                                num_types=self.num_types,
                                num_relations=self.num_relations).to(self.device)


        pytorch_total_params = sum(p.numel() for p in self.mblocks.parameters() if p.requires_grad)
        self.log.info('gnn parameters is: '+ str(pytorch_total_params)) 
        self.point2face = Point2FaceLayer(n_hid, hirechy_layer, 9, attention_head=3).to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.point2face.parameters() if p.requires_grad)
        self.log.info('face_atten parameters is: ' + str(pytorch_total_params)) 
        self.classifier = HierachyClassify(n_hid, self.num_label, hirechy_layer).to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        self.log.info('classifer parameters is: ' + str(pytorch_total_params)) 


        if self.loadPath!=None:
            self.log.info("load pre-trained model: "+ self.loadPath)
            self.model = torch.load(self.loadPath)
            self.mblocks = self.model[0]
            self.point2face = self.model[1]
            self.classifier = self.model[2]
        else:
            self.model = nn.Sequential(self.mblocks, self.point2face,  self.classifier)
            
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.loss_type = loss_type
        if loss_type == 'KLD':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        elif loss_type == 'focal':
            tll = set(tlabels)
            tll_div = [1 - (tlabels.count(i))/len(tlabels) for i in tll]
            self.criterion =  focal_loss(tll_div, len(tll))
        self.lr_schedual_type = lr_schedual_type
        self.best_acc = -1



    
    
    def train(self, iterations, folder=0, eta_min = 1e-6):
        if self.lr_schedual_type == 'Adaptive':
            self.log.info(' we choose a adaptive type')
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.9, patience=30,
                                                                        min_lr=1e-6, verbose=True)
        elif self.lr_schedual_type  == 'Cycle':
            self.log.info('we choose a cycle type')
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = iterations)
        elif self.lr_schedual_type == 'Cosine':
            self.log.info('we choose a Cosine type')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100, eta_min=1e-6)
        else:
            self.log.info('we choose a simple step type')
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=iterations/50, gamma=0.9)



        self.model.train()
        traindata = []
        assert folder >=0 and folder <=9
        for i in range(10):
            if i != folder:
                traindata += self.folders[i]
        
        
        train_losses = []
        torch.cuda.empty_cache()
        scount = 0
        t_count = 0
        sloss = 0.0
        record_loss = 0.0

        for step in range(iterations):
            for hgraph, ylabels, not_use_node_labes, file, face_area in traindata:
                node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict, nodefeature_location, face_indices = \
                    to_torch(hgraph.node_feature, hgraph.edge_list, hgraph)

                if torch.sum(torch.isnan(node_feature)) > 0:
                    continue

                node_rep, attention_edge_weights = self.mblocks.forward(node_feature.to(self.device), node_type.to(self.device), edge_weight.to(self.device), edge_index.to(self.device), edge_type.to(self.device))
                sourceMeshIds = torch.tensor(list(range(nodefeature_location['sn'][0],
                                                        nodefeature_location['sn'][0] + nodefeature_location['sn'][1])))
                source_to_cluster = edge_index.transpose(1, 0)[edge_type == edge_dict['s_c']]
                source_to_cluster_weight = attention_edge_weights[edge_type == edge_dict['s_c']]
                source_to_cluster_weightIds = source_to_cluster_weight[source_to_cluster[:, 0].sort()[1]]
                sourcetoClusterMeshIds = source_to_cluster[source_to_cluster[:, 0].sort()[1]][:, 1]
                source_to_global = edge_index.transpose(1, 0)[edge_type == edge_dict['s_g']]
                source_to_global_weight = attention_edge_weights[edge_type == edge_dict['s_g']]
                source_to_global_weightIds = source_to_global_weight[source_to_global[:, 0].sort()[1]]
                sourcetoGlobalMeshids = source_to_global[source_to_global[:, 0].sort()[1]][:, 1]

                if self.beginwriter == False:
                    self.beginwriter = True
                    self.writer.add_graph(self.mblocks, ( node_feature.to(self.device), node_type.to(self.device), edge_weight.to(self.device),
                    edge_index.to(self.device), edge_type.to(self.device)))
                vnode_rep = torch.hstack((node_rep[sourceMeshIds.to(self.device)],
                                         node_rep[sourcetoClusterMeshIds.to(self.device)] * source_to_cluster_weightIds[:, None] ,
                                         node_rep[sourcetoGlobalMeshids.to(self.device)] * source_to_global_weightIds[:, None]))
                node_rep = self.point2face(vnode_rep, face_indices)
                res = self.classifier.forward(node_rep)


                if self.loss_type == 'KLD':
                    loss = self.criterion(res, torch.FloatTensor(ylabels).to(self.device))
                elif self.loss_type == 'focal':
                    loss = self.criterion(res, torch.LongTensor(ylabels.argsort()[:, -1].reshape(-1).tolist()[0]).to(self.device), weight= torch.tensor(face_area).to(self.device))

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                loss.backward()
                self.optimizer.step()
                sloss += loss.item()
                record_loss += loss.item()

                self.log.info('The face area focal loss of '+file.split('/')[-1] + "th mesh:"+ str(loss.item()))
                if scount%5==0:
                    self.writer.add_scalar(str(folder) +'/training loss',
                                      sloss / 10,
                                      scount)
                    sloss = 0.0
                    lr = [group['lr'] for group in self.optimizer.param_groups]
                    self.writer.add_scalar(str(folder) +'/training_learningrate',
                                           lr[0],
                                           scount)


                train_losses += [loss.cpu().detach().tolist()]
                del res, loss
                scount+=1


            if self.lr_schedual_type == 'Adaptive':
                self.scheduler.step(record_loss)
                record_loss = 0.0
            elif self.lr_schedual_type == 'Cosine':
                self.scheduler.step(step)
            else:
                self.scheduler.step()


            if step % self.test_num == 0:
                self.log.info('test in the end of step: '+ str(step))
                self.test( t_count, folder=folder)
                t_count+=1

            if step % 100 == 0:
                error_label = self.classifier.forward(node_rep).argsort()[:, -1].reshape(-1) == torch.LongTensor(
                    ylabels.argsort()[:, -1].reshape(-1).tolist()[0]).to(self.device)
                face_center = torch.sum(node_feature[sourceMeshIds, :3][face_indices], dim=1)
                self.writer.add_embedding(face_center, metadata=error_label,
                                          global_step=step, tag=str(folder) +'/'+str(step)+'_'+'face_center')
                self.writer.add_histogram(str(folder) +'/input feature', node_feature, step)




    def test(self,t_count, folder = 0):
        self.model.eval()
        test_data = self.folders[folder]
        self.log.info('test data '+str(len(test_data)))
    
        
        label_acc = []
        error_label_save = None
        points_save = None 
        face_indices_save = None 
        
        
        for hgraph, ylabels, node_use_labels, file, face_area in test_data:
            with torch.no_grad():
                node_feature, node_type, edge_weight, edge_index, edge_type, node_dict, edge_dict,nodefeature_location, face_indices = \
                    to_torch(hgraph.node_feature, hgraph.edge_list, hgraph)
                if torch.sum(torch.isnan(node_feature)) > 0:
                    continue
                node_rep, attention_edge_weights = self.mblocks.forward(
                                                node_feature.to(self.device),
                                                node_type.to(self.device),
                                                edge_weight.to(self.device),
                                                edge_index.to(self.device),
                                                edge_type.to(self.device)
                                            )

                sourceMeshIds = torch.tensor(list(range(nodefeature_location['sn'][0],
                                                        nodefeature_location['sn'][0] + nodefeature_location['sn'][1])))
                source_to_cluster = edge_index.transpose(1, 0)[edge_type == edge_dict['s_c']]
                source_to_cluster_weight = attention_edge_weights[edge_type == edge_dict['s_c']]
                source_to_cluster_weightIds = source_to_cluster_weight[source_to_cluster[:, 0].sort()[1]]
                sourcetoClusterMeshIds = source_to_cluster[source_to_cluster[:, 0].sort()[1]][:, 1]

                source_to_global = edge_index.transpose(1, 0)[edge_type == edge_dict['s_g']]
                source_to_global_weight = attention_edge_weights[edge_type == edge_dict['s_g']]
                source_to_global_weightIds = source_to_global_weight[source_to_global[:, 0].sort()[1]]
                sourcetoGlobalMeshids = source_to_global[source_to_global[:, 0].sort()[1]][:, 1]

                points = node_feature[sourceMeshIds.to(self.device), :3]

                source_node_rep = torch.hstack((node_rep[sourceMeshIds.to(self.device)],
                                         node_rep[sourcetoClusterMeshIds.to(self.device)] * source_to_cluster_weightIds[:, None] ,
                                         node_rep[sourcetoGlobalMeshids.to(self.device)] * source_to_global_weightIds[:, None]))


                node_rep = self.point2face(source_node_rep, face_indices)
                res = self.classifier.forward(node_rep)
                error_label = res.argsort()[:, -1].reshape(-1) == torch.LongTensor(
                    ylabels.argsort()[:, -1].reshape(-1).tolist()[0]).to(self.device)
                face_area = torch.tensor(face_area).to(self.device)
                label_acc.append([torch.sum(error_label*face_area).tolist(), face_area.sum().tolist()])

                self.log.info(file.split('/')[-1]+"th mesh's test accuracy: " + str(torch.sum(error_label*face_area).tolist()/ face_area.sum()))
                error_label_save = error_label
                points_save = points
                face_indices_save = face_indices
                
        label_acc = np.array(label_acc)
        label_acc = np.sum(label_acc[:,0]) / np.sum(label_acc[:,1])
        self.log.info("label_mac acc is: " + str(label_acc)) 
        self.writer.add_scalar(str(folder) +'/label_mac', label_acc, t_count)

        self.log.info('best label accuracy is:' + str(self.best_acc)) 
        if label_acc > self.best_acc:
            self.best_acc = label_acc
            save_path = os.path.join(self.writerDir, 'best_model'+self.writerDir.replace('/', '_'))
            torch.save(self.model, save_path)

            self.log.info('UPDATE Best Model!!!')
            error_label = error_label_save 
            error_node_labels_true = torch.unique(face_indices[torch.where(error_label==False)[0]].reshape(-1))
            error_node_labels = torch.BoolTensor((len(sourceMeshIds))).to(self.device)
            error_node_labels[:] = False
            error_node_labels[error_node_labels_true] = True
            
            points = points_save
            face_indices = face_indices_save
            face_center = torch.sum(points[face_indices], dim=1)
            faces = torch.tensor(face_indices).to(self.device)
            self.writer.add_embedding(points,
                                      metadata=error_node_labels,
                                      global_step=t_count,tag=str(folder)+'/test')
            self.writer.add_embedding(face_center,
                                      metadata=self.classifier.forward(node_rep).argsort()[:, -1].reshape(-1),
                                      global_step=t_count, tag=str(folder)+'test_predict')
            self.writer.add_embedding(face_center,
                                      metadata=torch.LongTensor(
                ylabels.argsort()[:, -1].reshape(-1).tolist()[0]).to(self.device),
                                      global_step=t_count, tag=str(folder)+'test_real')

            self.writer.add_histogram(str(folder)+'/input node feature', node_feature, t_count)
            self.writer.add_histogram(str(folder)+'/output point feature', source_node_rep, t_count)
            self.writer.add_histogram(str(folder)+'/output face feature', node_rep, t_count)
            del error_label


    def closewrite(self):
        self.writer.close()







def parseParam():
    class LossTYPE(Enum):
        focal = 'focal'
        KLD = 'KLD'
        def __str__(self):
            return self.value
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-heterodir', type=str, required=True, help='Please input the heterodir.')
    parser.add_argument('-writerdir', type=str, required=True, help='Please input the writedir.')
    parser.add_argument('-losstype', type=LossTYPE, choices=list(LossTYPE), required=True, help='Please input the loss type.')
    parser.add_argument('-mask', nargs='+', default=[], type=int, help='mask features')
    parser.add_argument('-name', type=str, default="Anonymous_", help="please input the name of task")
    parser.add_argument('-device', type=int, default=0,  help="Please input the gpu id of using device.")
    parser.add_argument('-n_hid', type=int, default=400, required=True, help="Please input the dimension of features.")
    parser.add_argument('-n_layers', type=int, default=5, required=True, help="Please input the dimension of features.")
    parser.add_argument('-load_Path', type=str, default=None, help="Please input the path of pre-trained model.")
    args = parser.parse_args()
    return args



if __name__=='__main__':
    log = Logger(sys.path[-1])
    args = parseParam()
    log.info('using ' + args.heterodir+' as input dir.')
    log.info('using ' + args.writerdir+' as writer dir.')
    log.info('using ' + str(args.losstype)+' as losstype.')
    log.info('using ' + str(args.name)+' as name.')
    log.info('using ' + str(args.device) + ' as gpu id.')
    log.info('using ' + str(args.n_hid) + ' as hidder feature dim .')
    log.info('using ' + str(args.n_layers) + ' as the layers.')
    log.info('using ' + str(args.load_Path) + ' as pretrain model.')

    transformer = MeshTransformer(
                                            args,
                                            args.heterodir,
                                            args.writerdir,
                                            loss_type=str(args.losstype),
                                            lr_schedual_type='Step',
                                            task_name=args.name,
                                            device=args.device,
                                            n_hid=args.n_hid,
                                            n_layers=args.n_layers, 
                                            load_Path=args.load_Path
                                        )
    for i in range(10):
        transformer.train(1000, folder=i)











 
