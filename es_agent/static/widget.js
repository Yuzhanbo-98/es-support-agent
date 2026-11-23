/* 西语客服聊天插件：在任何网页（比如 Shopify 商店主题）里加一行
 *   <script src="https://你的服务地址/widget.js" defer></script>
 * 就会在右下角出现聊天按钮。接口地址自动取这个脚本自己的域名。
 * 卖家在后台同意/拒绝工单后，顾客这边的聊天窗口会自动收到西语通知（每 5 秒查一次）。
 */
(function () {
  if (window.__esAgentLoaded) return;
  window.__esAgentLoaded = true;

  var script = document.currentScript;
  var API = script ? new URL(script.src).origin : '';
  var KEY = 'esAgentCid';
  var cid = null;
  var shown = 0; // 已显示的服务器端消息条数
  try { cid = localStorage.getItem(KEY); } catch (e) {}

  var css = '' +
    '#esa-btn{position:fixed;right:20px;bottom:20px;z-index:2147483000;background:#111;color:#fff;border:0;' +
    'border-radius:24px;padding:12px 18px;cursor:pointer;font:600 15px/1 -apple-system,Segoe UI,Roboto,sans-serif;' +
    'box-shadow:0 4px 14px rgba(0,0,0,.2)}' +
    '#esa-box{position:fixed;right:20px;bottom:76px;z-index:2147483000;width:min(360px,calc(100vw - 32px));' +
    'height:min(500px,calc(100vh - 110px));background:#fff;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.2);' +
    'display:none;flex-direction:column;overflow:hidden;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;color:#222}' +
    '#esa-box.open{display:flex}' +
    '#esa-box .esa-top{background:#111;color:#fff;padding:12px 14px;font-weight:600;display:flex;justify-content:space-between}' +
    '#esa-box .esa-top button{background:none;border:0;color:#fff;font-size:18px;cursor:pointer;line-height:1}' +
    '#esa-log{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:8px}' +
    '.esa-b{max-width:82%;padding:8px 11px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}' +
    '.esa-me{align-self:flex-end;background:#111;color:#fff}' +
    '.esa-bot{align-self:flex-start;background:#f1f1f1}' +
    '.esa-seller{align-self:flex-start;background:#e7f6ee;border:1px solid #b7e4c7}' +
    '#esa-f{display:flex;border-top:1px solid #eee;margin:0}' +
    '#esa-q{flex:1;border:0;padding:12px;font:inherit;outline:none;min-width:0}' +
    '#esa-f button{border:0;background:#fff;padding:0 14px;cursor:pointer;font-weight:600;color:#111}';
  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  var btn = document.createElement('button');
  btn.id = 'esa-btn';
  btn.type = 'button';
  btn.textContent = '💬 ¿Te ayudamos?';
  var box = document.createElement('div');
  box.id = 'esa-box';
  box.innerHTML = '<div class="esa-top"><span>Atención al cliente</span><button type="button" aria-label="Cerrar">×</button></div>' +
    '<div id="esa-log"></div>' +
    '<form id="esa-f"><input id="esa-q" placeholder="Escribe tu mensaje..." autocomplete="off" maxlength="2000">' +
    '<button type="submit">Enviar</button></form>';
  document.body.appendChild(btn);
  document.body.appendChild(box);

  var log = box.querySelector('#esa-log');
  var input = box.querySelector('#esa-q');

  function add(text, who) {
    var d = document.createElement('div');
    d.className = 'esa-b esa-' + who;
    d.textContent = text;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  }

  function setCid(v) {
    cid = v;
    try { localStorage.setItem(KEY, v); } catch (e) {}
  }

  // 打开时恢复历史；之后定时取新消息（卖家审批结果）
  function sync(initial) {
    if (!cid) return Promise.resolve();
    return fetch(API + '/api/conversations/' + encodeURIComponent(cid) + '/messages')
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (msgs) {
        if (initial) {
          msgs.forEach(function (m) { add(m.content, m.role === 'customer' ? 'me' : (m.role === 'seller' ? 'seller' : 'bot')); });
        } else {
          msgs.slice(shown).forEach(function (m) { if (m.role === 'seller') add(m.content, 'seller'); });
        }
        shown = msgs.length;
      })
      .catch(function () {});
  }

  var started = false;
  function open() {
    box.classList.toggle('open');
    if (!box.classList.contains('open')) return;
    input.focus();
    if (started) return;
    started = true;
    sync(true).then(function () {
      if (!log.children.length) add('¡Hola! ¿En qué puedo ayudarte?', 'bot');
    });
    setInterval(function () { if (box.classList.contains('open')) sync(false); }, 5000);
  }
  btn.onclick = open;
  box.querySelector('.esa-top button').onclick = open;

  var busy = false;
  box.querySelector('#esa-f').onsubmit = function (ev) {
    ev.preventDefault();
    var text = input.value.trim();
    if (!text || busy) return;
    busy = true;
    input.value = '';
    add(text, 'me');
    var wait = add('…', 'bot');
    fetch(API + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, conversation_id: cid })
    })
      .then(function (r) {
        if (r.status === 429) throw new Error('rate');
        return r.json();
      })
      .then(function (data) {
        setCid(data.conversation_id);
        wait.textContent = data.reply;
        shown += 2;
      })
      .catch(function (e) {
        wait.textContent = e.message === 'rate'
          ? 'Has enviado muchos mensajes. Espera un momento, por favor.'
          : 'Error de conexión. Inténtalo de nuevo.';
      })
      .then(function () { busy = false; });
  };
})();
