<p align="center" width="100%">
<img src="assert/SeismicPretrainedModel.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://github.com/shenghanlin/' target='_blank'>Hanlin Sheng
    <sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xinming  Wu<sup>1,†,‡</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xu Si<sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Jintao Li<sup>1</sup></a>&emsp;
    </br>
    <a href='https://www.huawei.com/cn/' 
    target='_blank'>Sibo Zhang <sup>2</sup></a>&emsp;
    <a href='https://www.huawei.com/cn/' 
    target='_blank'>Xudong Duan <sup>2</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>
    University of Science and Technology of China&emsp;
    <sup>2</sup>
    Huawei&emsp;
    </br>
    <!-- <sup>*</sup> Equal Contribution&emsp; -->
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

-----------------

# 🌟 Seismic Foundation Model (SFM)

 As shown in this workflow figure, we test the Seismic Foundation Model's performance in segmentation tasks and regression tasks, specifically in classification (i.e. seismic facies), segmentaion (i.e. seismic geobody), signal processing (i.e. denoising), inversion (i.e. reflectivity estimation), and interpolation.

This is a PyTorch/GPU implementation of the paper [Seismic Foundation Model](https://arxiv.org/abs/2111.06377):
```
@Article{SeismicFoundationModel2023,
  author  = {Hanlin Sheng and Xinming wu and Xu Si and Jintao Li and Sibo Zhang and XuDong Duan},
  journal = {arXiv:...},
  title   = {Seismic Foundation Model},
  year    = {2023},
}
```

## 🌟 News
* **2023.8.7:** Github Repository Initialization (copy from Meta-Transformer). The paper and model will be release soon. ⌛⌛⌛


## &#x1F449; Pre-train & Fine-tune Code

* The pre-training instruction is in [PRETRAIN.md](SFM-Pretrain/README.md).

* The Fine-tuning instruction is in [FINETUNE.md](SFM-Finetune/README.md).


## :rocket: Model Zoo & Data Release

<!-- <details> -->
<summary> Open-source Pretrained Models </summary>
<br>
<div>

|    Model   |      Pretraining Size      |  Download |
|------------|:--------------------------:|:----------:|
| SFM-Base   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ef1xhsxytZRNjYiXJJGQAJEBZFmUiUhUuJxOyhILG88NRg?e=gGxUIb' target='_blank'>ckpt]    |  
| SFM-Base-512   |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/ES-iLYELZq1IiOor3LgDGPsBbRIXdt2wuyXeJfK-8FhM9w?e=5eURf2' target='_blank'>ckpt]    |  
| SFM-Large  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ec7ZqiAwdvtGpN5vasPnalwBXQe2qUPM_t9kdSjdkQeNIg?e=BmFlKU' target='_blank'>ckpt]    |
| SFM-Large-512  |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EXwVJmoAmRJFoenCvsIsolQBnyheFdjbejgryRj9esL2HA?e=gGsJaZ' target='_blank'>ckpt]    |    

<summary> Open-source Training & DownStream Fine-tune Task Data</summary>
<br>
<div>

|    Task   |      Size      |  Download |
|:------------------:|:--------------------------:|:----------:|
| PreTrain   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/Et8WP_voHfNMvx_kpR_iFVwBRpH3TgHsKPicCeRhXULn0g?e=f2cT2S' target='_blank'>DatFile]    |  
| Seismic Facies Classification   |         768 × 768          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/ElUKdIW6VhZOrekvngY7TqgBKYqgVfgC6fOg_vPdK8VYDA?e=xYrA0e' target='_blank'>DatFile]    |  
| Seismic GeoBody Identification  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/EvwMkQfKqJtOk6TP8U484yABSeCxjIL5gojWwqWSnMDeVg?e=NhbRWP' target='_blank'>DatFile]    |  
| Inversion (Reflectivity Estimation)  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/En2b7nDlY6BEn5tKXdcbi8oBTtO8CDRcir1IgGsnCYUeYw?e=dyTfnh' target='_blank'>DatFile]    |  
| Signal Processing (Denoise)   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/EnUGPcGo-hFFhrr2T4-wvSIB4KCQQJphdONXvaO1FOr_WA?e=rP057b' target='_blank'>DatFile]    |  
| Interpolation                 |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EWyYd0lXhfxOgffJIz5ICEUBRB_IqkbPoF1PQttUAfDLaQ?e=lR9qre' target='_blank'>DatFile]    |

  

<br>
<div>
# License
This project is released under the [MIT license](LICENSE).

