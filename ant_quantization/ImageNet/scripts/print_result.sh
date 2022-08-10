
vgg16_int=`tail -n 1 ./log/eval_vgg16_int.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`
vgg16_ip=`tail -n 1 ./log/eval_vgg16_ip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`
vgg16_fip=`tail -n 1 ./log/eval_vgg16_fip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`
vgg16_ip_f=`tail -n 1 ./log/eval_vgg16_ip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`
vgg16_fip_f=`tail -n 1 ./log/eval_vgg16_fip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`
vgg16_ant48=`tail -n 1 ./log/eval_vgg16_ant48.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'`

r18_int=`tail -n 1 ./log/eval_resnet18_int.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r18_ip=`tail -n 1 ./log/eval_resnet18_ip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r18_fip=`tail -n 1 ./log/eval_resnet18_fip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r18_ip_f=`tail -n 1 ./log/eval_resnet18_ip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r18_fip_f=`tail -n 1 ./log/eval_resnet18_fip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r18_ant48=`tail -n 1 ./log/eval_resnet18_ant48.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 

r50_int=`tail -n 1 ./log/eval_resnet50_int.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r50_ip=`tail -n 1 ./log/eval_resnet50_ip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r50_fip=`tail -n 1 ./log/eval_resnet50_fip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r50_ip_f=`tail -n 1 ./log/eval_resnet50_ip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r50_fip_f=`tail -n 1 ./log/eval_resnet50_fip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
r50_ant48=`tail -n 1 ./log/eval_resnet50_ant48.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 

incepv3_int=`tail -n 1 ./log/eval_inceptionv3_int.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
incepv3_ip=`tail -n 1 ./log/eval_inceptionv3_ip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
incepv3_fip=`tail -n 1 ./log/eval_inceptionv3_fip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
incepv3_ip_f=`tail -n 1 ./log/eval_inceptionv3_ip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
incepv3_fip_f=`tail -n 1 ./log/eval_inceptionv3_fip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
incepv3_ant48=`tail -n 1 ./log/eval_inceptionv3_ant48.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 

vit_int=`tail -n 1 ./log/eval_vit_int.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
vit_ip=`tail -n 1 ./log/eval_vit_ip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
vit_fip=`tail -n 1 ./log/eval_vit_fip.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
vit_ip_f=`tail -n 1 ./log/eval_vit_ip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
vit_fip_f=`tail -n 1 ./log/eval_vit_fip_f.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 
vit_ant48=`tail -n 1 ./log/eval_vit_ant48.log | grep "Final accuracy:" | awk -F" " '{ print $NF}'` 

echo " "
echo "| Network | Int  | IP | FIP | IP-F | FIP-F | ANT4-8 |"
echo "| :----:| :----: | :----: | :----: | :----: | :----: | :----: |"
echo "| VGG16 |" $vgg16_int"% |" $vgg16_ip"% |" $vgg16_fip"% |" $vgg16_ip_f"% |" $vgg16_fip_f"% |" $vgg16_ant48"% |"
echo "| ResNet18 |" $r18_int"% |" $r18_ip"% |" $r18_fip"% |" $r18_ip_f"% |" $r18_fip_f"% |" $r18_ant48"% |"
echo "| ResNet50 |" $r50_int"% |" $r50_ip"% |" $r50_fip"% |" $r50_ip_f"% |" $r50_fip_f"% |" $r50_ant48"% |"
echo "| InceptionV3 |" $incepv3_int"% |" $incepv3_ip"% |" $incepv3_fip"% |" $incepv3_ip_f"% |" $incepv3_fip_f"% |" $incepv3_ant48"% |"
echo "| ViT |" $vit_int"% |" $vit_ip"% |" $vit_fip"% |" $vit_ip_f"% |" $vit_fip_f"% |" $vit_ant48"% |"

