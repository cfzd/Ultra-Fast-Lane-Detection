
from data.dataloader import get_test_loader
from evaluation.tusimple.lane import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os, json, torch, scipy
import numpy as np
import platform

def generate_lines(out, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for j in range(out.shape[0]):
        out_j = out[j].data.cpu().numpy()
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        name = names[j]

        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            fp.write(
                                '%d %d ' % (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1))
                    fp.write('\n')

def run_test(net, data_root, exp_name, work_dir, griding_num, use_aux,distributed, batch_size=8):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, 'CULane', distributed)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2 and use_aux:
            out, seg_out = out

        generate_lines(out,imgs[0,0].shape,names,output_path,griding_num,localization_type = 'rel',flip_updown = True)
def generate_tusimple_lines(out,shape,griding_num,localization_type='rel'):

    out = out.data.cpu().numpy()
    out_loc = np.argmax(out,axis=0)

    if localization_type == 'rel':
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num)
        idx = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)

        loc[out_loc == griding_num] = griding_num
        out_loc = loc
    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:,i]
        lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1))) if loc != griding_num else -2 for loc in out_i]
        lanes.append(lane)
    return lanes

def run_test_tusimple(net,data_root,work_dir,exp_name,griding_num,use_aux, distributed,batch_size = 8):
    output_path = os.path.join(work_dir,exp_name+'.%d.txt'% get_rank())
    fp = open(output_path,'w')
    loader = get_test_loader(batch_size,data_root,'Tusimple', distributed)
    for i,data in enumerate(dist_tqdm(loader)):
        imgs,names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2 and use_aux:
            out = out[0]
        for i,name in enumerate(names):
            tmp_dict = {}
            tmp_dict['lanes'] = generate_tusimple_lines(out[i],imgs[0,0].shape,griding_num)
            tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
             270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 
             590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            tmp_dict['raw_file'] = name
            tmp_dict['run_time'] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str+'\n')
    fp.close()

def combine_tusimple_test(work_dir,exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)
    

def eval_lane(net, dataset, data_root, work_dir, griding_num, use_aux, distributed):
    net.eval()
    if dataset == 'CULane':
        run_test(net,data_root, 'culane_eval_tmp', work_dir, griding_num, use_aux, distributed)
        synchronize()   # wait for all results
        if is_main_process():
            res = call_culane_eval(data_root, 'culane_eval_tmp', work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
            P = TP * 1.0/(TP + FP)
            R = TP * 1.0/(TP + FN)
            F = 2*P*R/(P + R)
            dist_print(F)
        synchronize()

    elif dataset == 'Tusimple':
        exp_name = 'tusimple_eval_tmp'
        run_test_tusimple(net, data_root, work_dir, exp_name, griding_num, use_aux, distributed)
        synchronize()  # wait for all results
        if is_main_process():
            combine_tusimple_test(work_dir,exp_name)
            res = LaneEval.bench_one_submit(os.path.join(work_dir,exp_name + '.txt'),os.path.join(data_root,'test_label.json'))
            res = json.loads(res)
            for r in res:
                dist_print(r['name'], r['value'])
        synchronize()


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval(data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=30
    iou=0.5;  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir,'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir,'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir,'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir,'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir,'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir,'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir,'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir,'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0 = os.path.join(output_path,'txt','out0_normal.txt')
    out1=os.path.join(output_path,'txt','out1_crowd.txt')
    out2=os.path.join(output_path,'txt','out2_hlight.txt')
    out3=os.path.join(output_path,'txt','out3_shadow.txt')
    out4=os.path.join(output_path,'txt','out4_noline.txt')
    out5=os.path.join(output_path,'txt','out5_arrow.txt')
    out6=os.path.join(output_path,'txt','out6_curve.txt')
    out7=os.path.join(output_path,'txt','out7_cross.txt')
    out8=os.path.join(output_path,'txt','out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd']= read_helper(out1)
    res_all['res_night']= read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow']= read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve']= read_helper(out6)
    res_all['res_cross']= read_helper(out7)
    return res_all