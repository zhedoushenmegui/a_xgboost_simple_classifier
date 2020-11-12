package dev;

import com.alibaba.fastjson.JSONObject;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


/**
 * @author lemon
 * @date 2020/11/12
 * 参考: https://zhuanlan.zhihu.com/p/53729084
 */
public class PmmlRunClassifier {
    static final Logger logger = LoggerFactory.getLogger(PmmlRunClassifier.class);
    
    Evaluator evaluator;
    List<? extends InputField> inputFields;
    List<? extends TargetField> targetFields;
    String targetKey;
    
    public static final String pmmlPath = "lab/output/sample/xgb.pmml";
    public static final String fileName = "lab/output/sample/sample.txt";
    
    public void initModel() throws IOException, SAXException, JAXBException {
        File file = new File(pmmlPath);
        evaluator = new LoadingModelEvaluatorBuilder().load(file).build();
        evaluator.verify();
        // 获取模型定义的特征
        inputFields = evaluator.getInputFields();
        print("模型的特征是：", inputFields);
        // 获取模型定义的目标名称
        targetFields = evaluator.getTargetFields();
        targetKey = targetFields.get(0).getName().toString();
        print("目标字段是：", targetFields);
    }
    
    //定义一个实用函数，就是python中的print函数，没别的意思
    public static void print(Object... args){
        Arrays.stream(args).forEach(System.out::print);
        System.out.println("");
    }
    
    // 传入的参数是一个json，字段要求和模型的字段保持一致
    public Integer predict(JSONObject feature){
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for(InputField inputField: inputFields){
            FieldName inputName = inputField.getName();
            String name = inputName.getValue();
            Object rawValue = feature.getDoubleValue(name);
            FieldValue inputValue = inputField.prepare(rawValue);
            arguments.put(inputName, inputValue);
        }
        // 得到特征数据后就是预测了
        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        Map<String, ?> resultRecord = EvaluatorUtil.decodeAll(results);
        
        return (Integer) resultRecord.get(targetKey);
    }
    
    public static void main(String[] argv) {
        PmmlRunClassifier prc = new PmmlRunClassifier();
        try {
            prc.initModel();
        } catch (IOException | SAXException | JAXBException e) {
            e.printStackTrace();
            print("> failed to init");
            return;
        }
        ////
        File file = new File(fileName);
        List<String> brr = new ArrayList<>();
        try(BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String tempStr;
            while ((tempStr = reader.readLine()) != null) {
                brr.add(tempStr);
            }
        } catch (IOException e) {
            e.printStackTrace();
            print("> failed to load data");
            return;
        }
        ////
        print(brr.size());
        for(String line: brr) {
            String[] arr = line.split("#");
            double label = Double.parseDouble(arr[0]);
            
            JSONObject obj = JSONObject.parseObject(arr[1].trim());
            Integer pred = prc.predict(obj);
            print(String.format("python_pred:%s\t java_pred:%s\torigin:%s", label>.5?1:0, pred, obj.getString(prc.targetKey)));
            /// todo 判断预测是否一致
        }
    }
}
