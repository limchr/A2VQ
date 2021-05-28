function draw_everything(){
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'rgba(255, 255, 255, 1)';
  ctx.fillRect(0, 0, c.width, c.height);

  ctx2.clearRect(0, 0, c.width, c.height);

  draw_imgs();
  draw_selections();
}

function draw_imgs(){
  for (var i=0;i<Object.keys(samples).length;++i) {
    var img = new Image;
    img.src = 'data:image/png;base64,'+samples[i].thumb;
    img.i = i;
    img.onload = function() {
      if(this.height > this.width) {
        rh = dim_img[0]
        rw = Math.floor(this.width/this.height * dim_img[0])
      } else {
        rw = dim_img[0]
        rh = Math.floor(this.height/this.width * dim_img[0])
      }
      ctx.drawImage(this, samples[this.i].embedding[0]*(dim_canvas[0]-rw)+0,samples[this.i].embedding[1]*(dim_canvas[1]-rh)+0,rw,rh);
    };

  }
}

function draw_selections(){
    ctx2.clearRect(0, 0, c.width, c.height);
    ctx2.fillStyle = 'rgba(77, 100, 141, .5)';
    for (var s=0;s<selections.length;++s){
      ctx2.fillRect(selections[s][0],selections[s][2],(selections[s][1]-selections[s][0]),(selections[s][3]-selections[s][2]));
    }
}
function draw_selection(x,y,w,h){
    ctx2.fillStyle = 'rgba(40, 54, 85, .5)';
    ctx2.fillRect(x,y,w,h);
}

