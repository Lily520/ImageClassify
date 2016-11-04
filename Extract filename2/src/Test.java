/**
 * Created by Administrator on 2016/5/13.
 */

import java.io.*;

public class Test {
    public static void main(String[] args) {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null; //用于包装InputStreamReader,提高处理性能。因为BufferedReader有缓冲的，而InputStreamReader没有。
        try {
            String str = "";
            String path = "";
            String path2 = "";

            fis = new FileInputStream("C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101.txt");// FileInputStream
            // 从文件系统中的某个文件中获取字节
            isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
            br = new BufferedReader(isr);// 从字符输入流中读取文件中的内容,封装了一个new InputStreamReader的对象
            while ((str = br.readLine()) != null) {
                path="C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\"+str;
                path2 =  "C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\"+str+".txt" ;
                File file=new File(path);
                //File[] tempList = file.listFiles();
                String [] tempList = file.list();
                System.out.println("该目录下对象个数："+tempList.length);
                for (int i = 0; i < tempList.length; i++) {
                    // if (tempList[i].isFile()) {
                    //System.out.println("文 件："+tempList[i]);
                    Test.writeToFile(path2,tempList[i].toString()+"\r\n");

                    // }
                    //if (tempList[i].isDirectory()) {
                    //  System.out.println("文件夹："+tempList[i]);
                    //}

                }
            }
        } catch (FileNotFoundException e) {
            System.out.println("找不到指定文件");
        } catch (IOException e) {
            System.out.println("读取文件失败");
        } finally {
            try {
                br.close();
                isr.close();
                fis.close();
                // 关闭的时候最好按照先后顺序关闭最后开的先关闭所以先关s,再关n,最后关m
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void writeToFile(String filename,String content){

        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename, true));
            out.write(content);
            out.close();
        }
        catch (IOException e) { System.out.println("Error in writing to output file!"); }

        // System.out.println("outliers: " + nonmembers);
    }
}
