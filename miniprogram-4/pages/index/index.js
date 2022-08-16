Page({
    // data: {
    //     SHOW_TOP: true,
    //     canRecordStart: false,
    // },
    data: {
        tempFilePaths:'',
        sourceType: ['camera', 'album']
      },
    isSpeaking: false,
    accessToken: "",

 //头像点击处理事件，使用wx.showActionSheet()调用菜单栏
 uploadImg: function () {
    const that = this
    wx.showActionSheet({
      itemList: [/*'拍照',*/'相册'],
      itemColor: '',
      
      //成功时回调
      success: function (res) {
        if (!res.cancel) {
        /*
        res.tapIndex返回用户点击的按钮序号，从上到下的顺序，从0开始
        比如用户点击本例中的拍照就返回0，相册就返回1
        我们res.tapIndex的值传给chooseImage()
        */
       const checkeddata = true
       const that = this
       wx.chooseImage({
         //count表示一次可以选择多少照片
         count: 1,
         //sizeType所选的图片的尺寸，original原图，compressed压缩图
         sizeType: ['original', 'compressed'],
         //如果sourceType为camera则调用摄像头，为album时调用相册
         sourceType: ["album"],
         success(res) {
           // tempFilePath可以作为img标签的src属性显示图片
           console.log(res);
           const tempFilePaths = res.tempFilePaths
           //将选择到的图片缓存到本地storage中
           wx.setStorageSync('tempFilePaths', tempFilePaths)
           /*
           由于在我们选择图片后图片只是保存到storage中，所以我们需要调用一次   	        setHeader()方法来使页面上的头像更新
           */
        //   const tempFilePaths_ = wx.getStorageSync('tempFilePaths');
        //   if (tempFilePaths_) {
        //     that.setData({
        //       tempFilePaths: tempFilePaths_
        //     })
        //   } else {
        //     that.setData({
        //       tempFilePaths: '/images/camera.png'
        //     })
        //   }
           // wx.showToast({
           //   title: '设置成功',
           //   icon: 'none',
           // //   duration: 2000
           // })
           wx.showLoading({
               title: '识别中...',
           })
           
           var team_image = wx.getFileSystemManager().readFileSync(res.tempFilePaths[0], "base64")
           wx.request({
             url: 'http://127.0.0.1:5000/upload', //API地址，upload是我给路由起的名字，参照下面的python代码
             　　　　　 　　　　　method: "POST",
             header: {
               　　　　　　　　　'content-type': "application/x-www-form-urlencoded",
               　　　　　　　　},
             data: {image: team_image},//将数据传给后端
        
           success: function (res) {
               console.log(res.data);  //控制台输出返回数据  
               wx.hideLoading()
               wx.showModal({
   
                   title: '识别结果', 
                   confirmText: "确认",
                   cancelText:"取消",
                   content: res.data, 
                   success: function(res) { 
                   if (res.confirm) {
                   console.log('识别正确')
                   } else if (res.cancel) {
                   console.log('识别错误')
                   }
                   
                   }
                   
                   })     
             }
   
           })
           
         }
       })
      
   
        }
      },

 setHeader(){
    
 },


 chooseImg(tapIndex) {
    
 }

    })
  }
})
