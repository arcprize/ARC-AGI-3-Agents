class SidebarMenu(gr.HTML):        
    def __init__(
        self, menu_data, value=None, open=True, position="right", width=300, **kwargs
    ):

        html_template = """
        <div class="sidebar ${position} ${open ? 'open' : ''}" style="width: ${width}px; ${position}: -${width}px;">
            <button class="toggle-button" aria-label="Toggle Sidebar">
                <i data-lucide="chevron-right" class="toggle-icon"></i>
            </button>

            <div class="sidebar-content">
                <div class="sidebar-wrapper">
                    ${menu_data.map(item => {
                        if (item.type === 'group') {
                            return `
                            <div class="menu-group">
                                <div class="group-header">
                                    <div class="header-left">
                                        <i data-lucide="${item.icon}" class="icon" style="color: ${item.color || '#a9b1d6'};"></i> 
                                        <span class="label">${item.label}</span>
                                    </div>
                                    <i data-lucide="chevron-down" class="arrow icon"></i>
                                </div>
                                <div class="group-children">
                                    ${item.children.map(child => `
                                        <div class="menu-item ${value === child.id ? 'active' : ''}" data-id="${child.id}">
                                            <i data-lucide="${child.icon}" class="icon" style="color: ${child.color || '#a9b1d6'};"></i>
                                            <span class="label">${child.label}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>`;
                        } else {
                            return `
                            <div class="menu-item ${value === item.id ? 'active' : ''}" data-id="${item.id}">
                                <i data-lucide="${item.icon}" class="icon" style="color: ${item.color || '#a9b1d6'};"></i>
                                <span class="label">${item.label}</span>
                            </div>`;
                        }
                    }).join('')}
                </div>
            </div>
        </div>
        """

        css_template = """
            .sidebar {
                display: flex;
                flex-direction: column;
                position: fixed;
                top: 0;
                height: 100%;
                background-color: var(--background-fill-secondary);
                transform: translateX(0%);
                z-index: 1000;
                transition: transform 0.3s ease-in-out;
                border-right: 1px solid var(--border-color-primary);    
                color: var(--body-text-color);
                font-family: var(--font-sans-serif);    
            }

            .sidebar.open.left {
                transform: translateX(100%);
            }

            .sidebar.open.right {
                transform: translateX(-100%);
            }

            .toggle-button {
                position: absolute;
                top: 20px;
                background: var(--background-fill-secondary);    
                border: 1px solid var(--border-color-primary);    
                cursor: pointer;
                padding: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 28px;
                height: 32px;
                z-index: 1001;
                transition: all 0.3s ease;
                border-radius: 0;    
            }
            .toggle-button * {
                pointer-events: none;
            }
            .toggle-button:hover {
                background: var(--background-fill-primary);    
                border-color: var(--color-accent);    
            }

            .toggle-icon {
                width: 20px;
                height: 20px;
                color: var(--body-text-color-subdued);
                transition: transform 0.3s ease-in-out;
                stroke-width: 2.5;    
            }

            .sidebar.open .toggle-icon {
                transform: rotate(180deg);
            }

            .sidebar.left .toggle-button {
                left: 100%;
                border-radius: 0 8px 8px 0;
            }

            .sidebar.right .toggle-button {
                right: 100%;
                border-radius: 8px 0 0 8px;
            }

            .sidebar-content {
                height: 100%;
                overflow-y: auto;
                overflow-x: hidden;
            }

            .sidebar-content::-webkit-scrollbar {
                width: 6px;
            }

            .sidebar-content::-webkit-scrollbar-thumb {
                background: var(--border-color-primary);
                border-radius: 10px;
            }

            .sidebar-wrapper {
                font-family: 'Inter', sans-serif;
                padding: 20px 15px;
            }

            .menu-item {
                display: flex;
                align-items: center;
                padding: 12px 15px;
                margin-bottom: 4px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s ease;
            }

            .menu-item:hover {
                background-color: var(--background-fill-primary);
                color: var(--body-text-color)
            }

            .menu-item.active {
                background-color: var(--color-accent) !important;
                color: white !important;
            }

            .group-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 15px;
                margin-bottom: 4px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                color: var(--body-text-color);
                ;
                transition: background-color 0.2s;
            }

            .group-header:hover {
                background-color: var(--background-fill-primary);
            }

            .header-left {
                display: flex;
                align-items: center;
            }

            .group-children {
                display: none;
                padding-left: 15px;
                margin-top: 4px;
            }

            .icon {
                margin-right: 12px;
                width: 18px;
                height: 18px;
                stroke: currentColor;
                stroke-width: 2.2;
                color: inherit;
            }

            .toggle-icon {
                width: 20px;
                height: 20px;
                color: var(--body-text-color-subdued);
                transition: transform 0.3s ease-in-out;
                stroke-width: 2.5;
            }

            .arrow {
                width: 14px;
                height: 14px;
                transition: transform 0.3s ease;
                color: var(--body-text-color-subdued);
                margin-right: 0;
            }

            @media (max-width: 768px) {
                .sidebar {
                    width: 100vw !important;
                }

                .sidebar.left {
                    left: -100vw !important;
                }

                .sidebar.right {
                    right: -100vw !important;
                }
            }
        """

        js_on_load = """
            let openFolders = new Set();

            function initLucide(retries = 8, delay = 200) {
                if (window.lucide) {
                    window.lucide.createIcons();
                    element.querySelectorAll('i[data-lucide]').forEach(i => {
                        const svg = i.querySelector('svg');
                        if (svg) {
                            let desiredColor = i.style.color || getComputedStyle(i).color;
                            if (desiredColor && 
                                desiredColor !== 'rgb(0,0,0)' && desiredColor !== '#000000' &&
                                desiredColor !== 'rgb(255,255,255)' && desiredColor !== '#ffffff') {
                                svg.style.color = desiredColor;
                                const shapes = svg.querySelectorAll('path, line, polyline, circle, rect, polygon');
                                shapes.forEach(shape => {
                                    shape.setAttribute('stroke', desiredColor);
                                    shape.style.stroke = desiredColor + ' !important';
                                });
                                if (svg.hasAttribute('fill') && svg.getAttribute('fill') === 'currentColor') {
                                    svg.setAttribute('fill', desiredColor);
                                }
                                svg.querySelectorAll('[fill="currentColor"]').forEach(el => {
                                    el.setAttribute('fill', desiredColor);
                                });
                            }
                        }
                    });
                } else if (retries > 0) {
                    setTimeout(() => initLucide(retries - 1, delay), delay);
                }
            }

            function applyFolderStates() {
                element.querySelectorAll('.group-header').forEach(header => {
                    const label = header.querySelector('.label')?.innerText.trim();
                    if (!label) return;
                    const children = header.nextElementSibling;
                    const arrow = header.querySelector('.arrow');
                    if (children && arrow) {
                        const isOpen = openFolders.has(label);
                        children.style.display = isOpen ? 'block' : 'none';
                        arrow.style.transform = isOpen ? 'rotate(180deg)' : 'rotate(0deg)';
                    }
                });
            }

            initLucide();
            applyFolderStates();

            const observer = new MutationObserver(() => {
                setTimeout(() => {
                    applyFolderStates();
                    initLucide();
                }, 50);
            });

            observer.observe(element, {
                childList: true,
                subtree: true,
                attributes: true,
                characterData: true
            });

            element.addEventListener('click', (e) => {
                if (e.target.closest('.toggle-button')) {
                    const sidebar = element.querySelector('.sidebar');
                    sidebar.classList.toggle('open');
                    trigger(sidebar.classList.contains('open') ? 'expand' : 'collapse');
                    return;
                }

                const header = e.target.closest('.group-header');
                if (header) {
                    const label = header.querySelector('.label')?.innerText.trim();
                    if (label) {
                        if (openFolders.has(label)) openFolders.delete(label);
                        else openFolders.add(label);
                    }
                    applyFolderStates();
                    return;
                }

                const item = e.target.closest('.menu-item');
                if (item) {
                    props.value = item.dataset.id;
                    trigger('change');

                    if (window.innerWidth <= 768) {
                        element.querySelector('.sidebar')?.classList.remove('open');
                    }

                    setTimeout(() => {
                        applyFolderStates();
                        initLucide();
                    }, 0);
                }
            });
        """

        super().__init__(
            value=value,
            menu_data=menu_data,
            open=open,
            position=position,
            width=width,
            html_template=html_template,
            css_template=css_template,
            apply_default_css=False,
            js_on_load=js_on_load,
            **kwargs
        )

    def api_info(self):
        return {"type": "string"}
