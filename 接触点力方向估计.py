import pcl
import os
import pcl.pcl_visualization
import numpy as np
import json
import vtk_visualizer.visualizercontrol
import vtk
from vtk_visualizer.renderwidget import RenderWidget
from vtk_visualizer.pointobject import *
import sys
from PyQt5.QtWidgets import *

def ReadPoints( filename):
    """加载STL文件"""
    arr1 = []
    with open(filename, 'r')as file:
        for goal_line in range(3, 6):
            file.seek(0, 0)
            for line_num, each_line in enumerate(file):
                if (line_num == goal_line):
                    arr1.append(each_line.split())
                    goal_line += 7
    for i in range(len(arr1)):
        arr1[i].remove(arr1[i][0])
    # print("点集点的个数%d" % len(arr1))
    for i in range(len(arr1)):
        for p in range(3):
            arr1[i][p] = float(arr1[i][p])
    # print(arr1)
    return arr1


# 传入点云对象
def points2pcd(points,strbit):
    """
        将点云数据保存成pcd文件
    :param points: 需要保存的点云数据
    :param strbit: 标志位，选取存放类型及路径
    :return:
    """

    if(strbit=='m'):
        PCD_FILE_PATH = os.path.join('./model/move_test.pcd')
    elif(strbit=='f'):
        PCD_FILE_PATH = os.path.join('./model/fix_test.pcd')

    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = len(points)

    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
        handle.write(string)
    handle.close()

def addpoint(arr1,maxCount):
    """
        三角片面上均匀采样
    :param arr1: 三角片面顶点坐标
    :param maxCount: 插入点个数
    :return: 返回为均匀采样三角片面上的三维点坐标
    """
    arr_last=[]
    arr = []
    length=len(arr1)
    middle=int(length/3)
    last=int(length*2/3)
    for i in range(middle):
        arr.append(arr1[0+i])
        arr.append(arr1[middle+i])
        arr.append(arr1[last+i])
    for j in range(middle):
        cx=(arr[3*j][0]+arr[1+3*j][0]+arr[2+3*j][0])/3
        cy =( arr[3*j][1] + arr[1+3*j][1] + arr[2+3*j][1])/3
        cz =( arr[3*j][2] + arr[1+3*j][2] + arr[2+3*j][2])/3
        for i in range(3):
            count=0
            index1=i+3*j
            index2=i+1+3*j
            if(i==2):
                index1=0+3*j
                index2=2+3*j
            while(count<maxCount):
                ab1 = arr[index1][0] - cx
                ab2 = arr[index1][1] - cy
                ab3 = arr[index1][2] - cz
                ac1 = arr[index2][0] - cx
                ac2 = arr[index2][1] - cy
                ac3 = arr[index2][2] - cz
                x = np.random.rand()
                y = np.random.rand()
                if (x + y > 1):
                    x1 = 1 - x
                    y1 = 1 - y
                else:
                    x1 = x
                    y1 = y
                pt=[cx+ab1*x1+ac1*y1,
                        cy+ab2*x1+ac2*y1,
                        cz+ab3*x1+ac3*y1]
                arr_last.append(pt)
                count+=1
    # print(arr_last)
    return arr_last


def STl2PCD(string):
    """
    STL转换成点云数据
    :param string: 标志位，m代表移动物体，f代表固定物体
    :return:
    """
    try:
        if ("m" == string[8]):
            point = ReadPoints(string)
            points=addpoint(point, 100)                  #重要，通过这个来设置采样个数
            points2pcd(points,'m')
            cloud = pcl.load('./model/move_test.pcd')
        elif ("f" == string[8]):
            point = ReadPoints(string)
            points = addpoint(point, 100)                #重要，通过这个来设置采样个数
            points2pcd(points, 'f')
            cloud = pcl.load('./model/fix_test.pcd')
        return cloud

    except Exception:
        print("name is wrong")
        return -1

def ExtractNormals(cloud):
    """
    计算点云所有点的公法线
    :param cloud:点云中所有点的坐标
    :return:    param[out] 法向量x分量
                param[out] 法向量y分量
                param[out] 法向量z分量
                param[out] 曲率
    """
    try:
        kd_tree=cloud.make_kdtree()   #创建KD-tree
        ne = cloud.make_NormalEstimation() #进行公法线估计
        ne.set_KSearch(50)     #选取K近邻
        ne.set_SearchMethod(kd_tree)  #选择搜索方法
        normals = ne.compute()   #进行计算
        # print('compute - end')
        # print(str(normals.size))
        # np.set_printoptions(threshold=np.inf)
        # for i,j in enumerate(normals.to_array()):
        #     print (i,j)
        return  normals
    except Exception:
        print("no cloud ")
        return -1

def ExtractRealPoint(pc_1,points_2):
    """
    匹配真实点
    由于点云生成我们是基于三角片面均匀采样，虽然理论上当我们maxCount设为很大，可以做到逼近真实物体，但是这会造成点云很大，当我们计算的时候会很耗时，
    我们采取合适的maxCount，而我们采取k近邻搜索，选择离关键接触点最近的数据作为我们所求局部特征的点
    :param pc_1: 点云中的点
    :param points_2: 关键接触点
    :return:
    """
    pc_2 = pcl.PointCloud(points_2)
    kd = pc_1.make_kdtree_flann()
    indices, sqr_distances = kd.nearest_k_search_for_cloud(pc_2, 1)
    list = []
    for i in range(pc_2.size):
        # print('index of the closest point in pc_1 to point %d in pc_2 is %d'
        #       % (i, indices[i, 0]))
        list.append(indices[i, 0])
        # print('the squared distance between these two points is %f'
        #       % sqr_distances[i, 0])
    print(list)
    pc_3 = pc_1.extract(list)
    return pc_3,list

def Viewer(cloud):
    """
    可视化点云及关键接触点，和公法线
    :param cloud:点云
    :return:
    """

    obj = VTKObject()
    obj.CreateFromArray(cloud[1].to_array())
    obj.AddNormals(cloud[2].to_array()[0: , 0: 3])
    obj.SetupPipelineHedgeHog(10)
    ren = vtk.vtkRenderer()
    ren.AddActor(obj.GetActor())
    obj2 = VTKObject()
    obj2.CreateFromArray(cloud[0].to_array())
    ren.AddActor(obj2.GetActor())
    obj3 = VTKObject()
    obj3.CreateFromArray(cloud[1].to_array())
    obj3.GetActor().GetProperty().SetColor(1, 0, 0.0)
    obj3.GetActor().GetProperty().SetPointSize(10)
    ren.AddActor(obj3.GetActor())
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.Initialize()
    iren.Start()

    # viewer = pcl.pcl_visualization.PCLVisualizering()
    # viewer.SetBackgroundColor(0.5, 0.5, 0.5)
    # color1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud[0], 0, 255, 0)
    # color2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud[1], 255, 0, 0)
    # print(cloud[1])
    # viewer.AddPointCloud_ColorHandler(cloud[0], color1, b'cloud1')
    # viewer.AddPointCloud_ColorHandler(cloud[1], color2, b'cloud2')
    # # viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 5, b'cloud1')
    # viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 5, b'cloud2')
    # viewer.AddCoordinateSystem(10.0)
    #
    # flag = True
    # while (flag):
    #     flag = not (viewer.WasStopped())
    #     viewer.SpinOnce()


def func(pointcloud,normal,points):
    """
    匹配对应点的局部特征
    :param pointcloud:点云
    :param normal:点云中所有点的公法线
    :param points:关键接触点
    :return:
    """
    pointcloud_extract, exact_list = ExtractRealPoint(pointcloud, points)
    k = 0
    normal_extract = []
    while (k != len(exact_list)):
        for i, j in enumerate(normal.to_array()):
            if (exact_list[k] == i):
                normal_extract.append(j.tolist())
                k += 1
                break
    string = normal_extract
    print("关键接触点的法线与曲率:")
    for i,j in enumerate(normal_extract):
        print("第%s关键接触点，对应的法线与曲率为:%s"%(i,j))
    normal_extract = np.array(normal_extract, dtype=np.float32)
    normal_extract = pcl.PointCloud_Normal(normal_extract)
    PointCloud_List = [pointcloud, pointcloud_extract, normal_extract]
    Viewer(PointCloud_List)
    return string


class RWtext(object):
    """
    记载关键接触点
    """
    def __init__(self):
        self.arr = []
        self.goal_line=1
        self.arr1 = []
        self.arr2 = []

    def ReadText(self, file):
        with open(file, 'r')as filename:
            filename.seek(0, 0)
            a = filename.readline()
            while (a != ""):
                if (a[0] == "第"):
                    a = filename.readline()
                    if (a[0] == "["):
                        self.arr.append(a.strip('\n'))
                        a = filename.readline()
                    elif (a[0] == "一"):
                        a = filename.readline()
                        a = a.strip('\n')
                        b = filename.readline()
                        if (b[0] == "二"):
                            c = filename.readline()
                            c = c.strip('\n')
                            self.arr.append(a[:-1] + "," + c[1:])
                            a = filename.readline()
                        else:
                            self.arr.append(a[:-1])
                            a = b
                            continue
                    elif (a[0] == "二"):
                        a = filename.readline()
                        a = a.strip('\n')
                        self.arr.append(a[:-1])
                        a = filename.readline()
                    else:
                        self.arr.append("[[0, 0, 0]]")
            for p in range(len(self.arr)):
                self.arr1.append([float(i) for i in self.arr[p].replace("[", "").replace("]", "").split(",")])
            for j in range(len(self.arr1)):
                self.arr2.append([self.arr1[j][i:i + 3] for i in range(0, len(self.arr1[j]), 3)])


    # def ReadText(self,file):
    #     with open(file, 'r')as filename:
    #         filename.seek(0, 0)
    #         a = filename.readline()
    #         while(a!=""):
    #             if(a[0]=="第"):
    #                 a = filename.readline()
    #                 if(a[0]=="["):
    #                     self.arr.append(a.strip('\n'))
    #                     a = filename.readline()
    #                 elif(a[0]=="一"):
    #                     a = filename.readline()
    #                     a=a.strip('\n')
    #                     b= filename.readline()
    #                     c = filename.readline()
    #                     c=c.strip('\n')
    #                     self.arr.append(a[:-1]+","+c[1:])
    #                     a = filename.readline()
    #                 elif (a[0] == "二"):
    #                     a = filename.readline()
    #                     a = a.strip('\n')
    #                     self.arr.append(a[:-1])
    #                     a = filename.readline()
    #                 else:
    #                     self.arr.append("[[0, 0, 0]]")
    #         for p in range(len(self.arr)):
    #             self.arr1.append([float(i) for i in self.arr[p].replace("[", "").replace("]", "").split(",")])
    #         for j in range(len(self.arr1)):
    #             self.arr2.append([self.arr1[j][i:i+3] for i in range(0, len(self.arr1[j]), 3)])

                # for j in range(int(len(self.arr1[i])/3)):
                #     self.arr1[i].append([self.arr1[i][3*j],self.arr1[i][3*j+1],self.arr1[i][3*j+2]])

    def WriteCondition(self):
        pass
    def WriteKeypoint(self):
        pass

    def Getpoint(self):
        return self.arr2




def main():
    """
    针对于过程中所有数据的局部特征分析
    :return:
    """
    rw = RWtext()
    rw.ReadText("key_point.txt")
    arr = rw.Getpoint()    # 获得关键接触点
    data=[]
    for i in range(len(arr)):
        print("第%d组数据力方向估算:"%i)
        points= np.array(arr[i], dtype=np.float32)

        flag = 1  #True为move，False为fix
        move_stl = "./model/model_thing/move" + str(i+1) + ".STL"
        fix_stl = "./model/model_thing/fix" + str(i+1) + ".STL"
        pointcloud_move=STl2PCD(move_stl)
        normal_move=ExtractNormals(pointcloud_move)
        if(-1==normal_move):
            print("normal_move is wrong ")
            return 0
        pointcloud_fix = STl2PCD(fix_stl)
        normal_fix=ExtractNormals(pointcloud_fix)
        if (-1 == normal_fix):
            print("normal_fix is wrong ")
            return 0
        if(flag):
            string=func(pointcloud_move,normal_move,points)
        else:
            string=func(pointcloud_fix,normal_fix,points)
        data.append({'normal': string,
                     })
        # with open("force_direction.json", "w")as file:
        #     file.write(json.dumps(data, ensure_ascii=False, indent=2))



def main2():
    """
    对于某一时刻的局部特征分析
    :return:
    """
    # # point1=[19.057145122071386, -22.169326253072537, -29.909623797468353]
    # # point2=[19.057145122071386, -22.169326253072537, -29.909623797468353]
    # point3 = [[(point1[i] + point20[i]) / 2 for i in range(len(point2))]]
    point3=[[6.106065411113209, 4.539157987785113, -29.97620429614166]]
    points = np.array(point3, dtype=np.float32)

    flag = 1  # True为move，False为fix
    move_stl = "./model/model_thing/move" + str( 302) + ".STL"
    fix_stl = "./model/model_thing/fix" + str(302) + ".STL"
    pointcloud_move = STl2PCD(move_stl)
    normal_move = ExtractNormals(pointcloud_move)
    if (-1 == normal_move):
        print("normal_move is wrong ")
        return 0
    pointcloud_fix = STl2PCD(fix_stl)
    normal_fix = ExtractNormals(pointcloud_fix)
    if (-1 == normal_fix):
        print("normal_fix is wrong ")
        return 0
    if (flag):
        string = func(pointcloud_move, normal_move, points)
    else:
        string = func(pointcloud_fix, normal_fix, points)
    print(string)


if __name__ == '__main__':
    main2()
