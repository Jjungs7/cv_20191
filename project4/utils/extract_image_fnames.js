realimages = [];
fakeimages = [];
image_amount = 70000;

function saveBlobFile(blob, fileName) {
    var reader = new FileReader();
    reader.onloadend = function () {
        var base64 = reader.result ;
        var link = document.createElement("a");

        link.setAttribute("href", base64);
        link.setAttribute("download", fileName);
        link.click();
    };
    reader.readAsDataURL(blob);
}

function saveTxtFile(data, filename, type="text/plain") {
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob) // IE10+
        window.navigator.msSaveOrOpenBlob(file, filename);
    else { // Others
        var a = document.createElement("a"),
                url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}

function getImage(url, fname) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url);
    xhr.responseType = "blob";
    xhr.onload = e => {
        saveBlobFile(xhr.response, fname);
    }
}

function httpGet(theUrl) {
    if (window.XMLHttpRequest)
    {// code for IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp=new XMLHttpRequest();
    }
    else
    {// code for IE6, IE5
        xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    xmlhttp.onreadystatechange=function()
    {
        if (xmlhttp.readyState==4 && xmlhttp.status==200)
        {
			let real_img_fname = '';
			let fake_img_fname = '';
			let text = xmlhttp.responseText;
			let idxOf = text.indexOf('&i1=realimages/');
			if(idxOf >= 0) text = text.substring(idxOf + 15);
			else text = text.substring(text.indexOf('&i1=fakeimages/') + 15);

			let amp = text.indexOf('&');
			if(idxOf >= 0) {
			    real_img_fname = text.substring(0, amp);
			    fake_img_fname = text.substring(amp + 15, text.indexOf('g">') + 1);
			}
            else {
                real_img_fname = text.substring(amp + 15, text.indexOf('g">') + 1);
                fake_img_fname = text.substring(0, amp);
            }
            realimages.push(real_img_fname + ' http://www.whichfaceisreal.com/realimages/' + real_img_fname);
            fakeimages.push(fake_img_fname + ' http://www.whichfaceisreal.com/fakeimages/' + fake_img_fname);

            if(realimages.length === image_amount) {
                saveTxtFile(realimages.join('\n'), 'proj4_realimglist.txt');
                saveTxtFile(fakeimages.join('\n'), 'proj4_fakeimglist.txt');
            }
        }
    }
    xmlhttp.open("GET", theUrl, false);
    xmlhttp.send();
}